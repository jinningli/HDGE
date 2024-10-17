import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

Epsilon = 1e-7

from dataset import visualize_tensor


def visualize_2d_tensor_and_save(tensor, file_path, x_label='X-axis', y_label='Y-axis', title='2D Tensor Scatter Plot'):
    # Ensure the tensor is 2D and has two columns for scatter plot
    assert tensor.ndimension() == 2 and tensor.size(1) == 2, "Tensor must be 2D with two columns for scatter plot"

    # Detach the tensor from the computation graph and convert to numpy for plotting
    tensor_np = tensor.detach().cpu().numpy()

    # Separate the columns into X and Y coordinates
    x_coords = tensor_np[:, 0]
    y_coords = tensor_np[:, 1]

    # Create a scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(x_coords, y_coords, color='blue', marker='o', s=8)

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Save plot to file
    plt.grid(True)
    plt.savefig(file_path, dpi=300)
    plt.close()


def save_tensor_to_text(tensor, filename):
    """
	Saves a 2D PyTorch tensor to a text file with each number formatted to .3f.

	Args:
	- tensor (torch.Tensor): 2D tensor to be saved.
	- filename (str): The file path where the tensor will be saved.
	"""
    # Ensure the tensor is 2D
    if tensor.dim() != 2:
        raise ValueError("Input tensor must be 2D")

    # Open the file for writing
    with open(filename, 'w') as f:
        for row in tensor:
            # Convert each row to a formatted string with 3 decimal places
            row_str = ' '.join([f"{num:.3f}" for num in row])
            f.write(row_str + '\n')


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.activation = activation

    def forward(self, adj, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


def normalize_adj(adj: torch.Tensor):
    adj_ = adj + torch.eye(adj.shape[0], device=adj.device)
    rowsum = torch.sum(adj_, dim=1)
    degree_mat_inv_sqrt = torch.diag(torch.pow(rowsum, -0.5))
    adj_normalized = torch.mm(torch.mm(degree_mat_inv_sqrt, adj_), degree_mat_inv_sqrt)
    return adj_normalized


class PolarEncoder(nn.Module):
    def __init__(self, init_adj, num_user, num_assertion, feature_dim, embedding_dim, device, hidden_dim=32):
        super(PolarEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_user = num_user
        self.num_assertion = num_assertion
        self.init_adj = init_adj  # constant, no diagonal
        self.init_adj.requires_grad_(False)
        self.init_adj_target = init_adj[:self.num_user, self.num_user:]  # constant
        self.base_gcn = GraphConv(self.feature_dim, self.hidden_dim, activation=F.relu).to(device)
        self.gcn_mean = GraphConv(self.hidden_dim, self.embedding_dim, activation=lambda x: x).to(device)
        self.gcn_logstddev = GraphConv(self.hidden_dim, self.embedding_dim, activation=lambda x: x).to(device)
        # Adj slice
        self.to_slice_index = {k: k for k in range(self.num_user + self.num_assertion)}
        self.to_origin_index = {k: k for k in range(self.num_user + self.num_assertion)}
        self.adj_sliced = self.init_adj
        self.origin_rows_to_keep = torch.ones(self.init_adj.shape[0], dtype=torch.bool, device=device)
        self.adj_sliced_norm = normalize_adj(self.init_adj)
        self.adj_sliced_target = self.init_adj_target
        self.sliced_num_user = self.num_user
        self.sliced_num_assertion = self.num_assertion

    def update_sliced_matrix(self, belief_mask, semi_supervision_keep=None, epsilon=1e-6):
        # belief_mask = belief_mask.detach() ########################
        full_mask = torch.ones(self.num_user + self.num_assertion, device=belief_mask.device)
        full_mask[self.num_user:] = belief_mask
        adj_masked = self.init_adj * full_mask.view(-1, 1) * full_mask.view(1, -1)
        row_sums = adj_masked.sum(dim=1)
        rows_to_keep = row_sums > epsilon
        if semi_supervision_keep is not None:
            rows_to_keep[semi_supervision_keep] = True  # Semi-supervision nodes always kept
        adj_sliced = adj_masked[rows_to_keep][:, rows_to_keep]
        indices_kept = torch.nonzero(rows_to_keep).squeeze()
        self.to_slice_index = {int(original_idx): sliced_idx for sliced_idx, original_idx in enumerate(indices_kept)}
        self.to_origin_index = {sliced_idx: int(original_idx) for sliced_idx, original_idx in enumerate(indices_kept)}
        self.adj_sliced = adj_sliced
        self.origin_rows_to_keep = rows_to_keep
        self.adj_sliced_norm = normalize_adj(self.adj_sliced)
        self.sliced_num_user = 0
        while self.to_origin_index[self.sliced_num_user] < self.num_user:
            self.sliced_num_user += 1
        self.sliced_num_assertion = self.adj_sliced.shape[0] - self.sliced_num_user
        self.adj_sliced_target = self.adj_sliced[:self.sliced_num_user, self.sliced_num_user:].detach()

    def encode(self, x, belief_mask=None):
        if belief_mask is not None:
            assert belief_mask.ndim == 1
            self.update_sliced_matrix(belief_mask)
        x = x[self.origin_rows_to_keep]
        hidden = self.base_gcn(self.adj_sliced_norm, x)
        self.mean = self.gcn_mean(self.adj_sliced_norm, hidden)
        self.logstd = self.gcn_logstddev(self.adj_sliced_norm, hidden)
        gaussian_noise = torch.randn(x.size(0), self.embedding_dim, device=x.device)
        sampled_z = F.relu(gaussian_noise * torch.exp(self.logstd) + self.mean)  # Non-Negative
        return sampled_z

    def decode(self, z):
        inner_prod = torch.matmul(z[:self.sliced_num_user], z[self.sliced_num_user:].t())
        # Output matrix: [U, T]
        return 2 * torch.sigmoid(inner_prod) - 1

    def forward(self, x, belief_mask=None):
        z = self.encode(x, belief_mask)
        return self.decode(z)


class BeliefEncoder(nn.Module):
    def __init__(self, init_adj, feature_dim, embedding_dim, device, hidden_dim=32):
        super(BeliefEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.init_adj = init_adj  # constant, no diagonal
        self.init_adj.requires_grad_(False)
        self.base_gcn = GraphConv(self.feature_dim, self.hidden_dim, activation=F.relu).to(device)
        self.gcn_mean = GraphConv(self.hidden_dim, self.embedding_dim, activation=lambda x: x).to(device)
        self.gcn_logstddev = GraphConv(self.hidden_dim, self.embedding_dim, activation=lambda x: x).to(device)
        self.init_adj_norm = normalize_adj(self.init_adj)
        self.init_adj_norm.requires_grad_(False)
        self.adj_target = self.init_adj + torch.eye(self.init_adj.shape[0], device=device)

    def encode(self, x):
        hidden = self.base_gcn(self.init_adj_norm, x)
        self.mean = self.gcn_mean(self.init_adj_norm, hidden)
        self.logstd = self.gcn_logstddev(self.init_adj_norm, hidden)
        gaussian_noise = torch.randn(x.size(0), self.embedding_dim, device=x.device)
        sampled_z = F.relu(gaussian_noise * torch.exp(self.logstd) + self.mean)  # Non-Negative
        return sampled_z

    def decode(self, z):
        return 2 * torch.sigmoid(torch.matmul(z, z.t())) - 1

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class ModelTrain:
    def __init__(self, dataset, args):
        # Inputs
        self.args = args
        self.dataset = dataset
        # Model
        self.belief_encoder = None
        self.polar_encoders = []
        # Output
        self.belief_embedding = None
        self.polar_embeddings = []

    def pos_weight(self, adj):
        pos_sum = adj[adj >= 0.5].sum() + Epsilon
        neg_adj = 1.0 - adj
        neg_sum = neg_adj[neg_adj > 0.5].sum() + Epsilon
        return float(neg_sum / pos_sum * self.args.pos_weight_lambda), pos_sum, neg_sum

    def get_pos_weight_vector(self, adj):
        weight_mask = adj.reshape(-1) >= 0.5
        weight_tensor = torch.ones(weight_mask.size(0)).to(self.args.device)
        pos_weight, pos_sum, neg_sum = self.pos_weight(adj)
        weight_tensor[weight_mask] = pos_weight
        return weight_tensor

    def bce_loss_norm(self, adj, pos_sum=None, neg_sum=None):
        if pos_sum is not None and neg_sum is not None:
            return adj.shape[0] * adj.shape[1] / float(neg_sum * (1.0 + self.args.pos_weight_lambda))
        else:
            pos_weight, pos_sum, neg_sum = self.pos_weight(adj)
            return adj.shape[0] * adj.shape[1] / float(neg_sum * (1.0 + self.args.pos_weight_lambda))

    def compute_semi_loss(self, emb, semi_adj_matrix, semi_units, semi_indexes, semi_index_mapping=None):
        if semi_index_mapping is not None:
            mapped_indexes = [semi_index_mapping[ind] for ind in semi_indexes]
            semi_indexes = mapped_indexes
        semi_units = semi_units.to(self.args.device)
        semi_adj_matrix = semi_adj_matrix.to(self.args.device)
        pred = torch.matmul(emb[semi_indexes], semi_units.t())
        pred = torch.sigmoid(pred)
        return self.bce_loss_norm(semi_adj_matrix) * F.binary_cross_entropy(
            pred.view(-1),
            semi_adj_matrix.view(-1),
            weight=self.get_pos_weight_vector(semi_adj_matrix))

    def train(self):
        # Inputs
        belief_feature = torch.tensor(self.dataset.belief_feature.toarray().astype(np.float32),
                                      device=self.args.device, requires_grad=False)
        polar_feature = torch.tensor(self.dataset.polar_feature.toarray().astype(np.float32),
                                     device=self.args.device, requires_grad=False)
        belief_matrix = torch.tensor(self.dataset.belief_matrix.toarray().astype(np.float32), device=self.args.device,
                                     requires_grad=False)
        polar_matrix = torch.tensor(self.dataset.polar_matrix.toarray().astype(np.float32), device=self.args.device,
                                    requires_grad=False)

        # Model
        self.belief_encoder = BeliefEncoder(
            init_adj=belief_matrix,
            feature_dim=self.dataset.belief_feature.shape[0],
            hidden_dim=self.args.hidden_dim,
            embedding_dim=self.args.belief_dim,
            device=self.args.device
        ).to(self.args.device)
        for _ in range(self.args.belief_dim):
            self.polar_encoders.append(PolarEncoder(
                init_adj=polar_matrix,
                num_user=self.dataset.num_user,
                num_assertion=self.dataset.num_assertion,
                feature_dim=self.dataset.polar_feature.shape[0],
                hidden_dim=self.args.hidden_dim,
                embedding_dim=self.args.polar_dim,
                device=self.args.device
            ).to(self.args.device))

        # Optimizer
        all_params = [{"params": self.belief_encoder.parameters(), "lr": self.args.learning_rate}]
        for k in range(self.args.belief_dim):
            all_params += [{"params": self.polar_encoders[k].parameters(), "lr": self.args.learning_rate}]
        optimizer = torch.optim.Adam(all_params, weight_decay=1e-4)

        # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

        def lr_cooldown(optimizer, cooldown_factor=self.args.lr_cooldown):
            optimizer.param_groups[0]["lr"] *= cooldown_factor  # The first one is for belief encoder

        for epoch in range(self.args.epochs):
            # Optimizer Cooldown
            if epoch == self.args.belief_warmup:
                lr_cooldown(optimizer)
                print(f"[Epoch {epoch}] Warmup ends. Learning rate cooldown for belief encoder.")
                print("Current Learning Rate: Belief Encoder={}, Polar Encoders=[{}]".format(
                    "{:.1e}".format(optimizer.param_groups[0]['lr']),
                    ", ".join(["{:.1e}".format(optimizer.param_groups[k]['lr']) for k in
                               range(1, len(optimizer.param_groups))])
                ))
            optimizer.zero_grad()

            # Belief Encoder
            loss = 0.0
            belief_semi_loss = None
            belief_emb = self.belief_encoder.encode(belief_feature)  # N x 7
            save_tensor_to_text(belief_emb.detach(), self.args.output_path / f"{epoch}.txt")  ################## DEBUG
            belief_A_pred = self.belief_encoder.decode(belief_emb)
            belief_recon_loss = self.bce_loss_norm(self.belief_encoder.adj_target) * F.binary_cross_entropy(
                belief_A_pred.view(-1), self.belief_encoder.adj_target.view(-1),
                weight=self.get_pos_weight_vector(self.belief_encoder.adj_target))
            loss = loss + belief_recon_loss
            if self.dataset.semi_variables["belief_semi_n"] > 0:
                belief_semi_loss = self.compute_semi_loss(
                    belief_emb,
                    semi_adj_matrix=self.dataset.semi_variables["belief_semi_adj_matrix"],
                    semi_units=self.dataset.semi_variables["belief_semi_units"],
                    semi_indexes=self.dataset.semi_variables["belief_semi_indexes"]
                )
                loss += belief_semi_loss
            belief_emb_softmax = F.softmax(belief_emb / self.args.temperature, dim=1)

            # Polar Encoders
            polar_recon_losses = []
            polar_semi_losses = []
            pos_weights = []
            if epoch >= self.args.belief_warmup:
                for k in range(self.args.belief_dim):
                    belief_mask = belief_emb_softmax[:, k]  # Belief Mask for Tweets [T,]
                    if self.dataset.semi_variables["polar_semi_n"][k] > 0:
                        self.polar_encoders[k].update_sliced_matrix(
                            belief_mask,
                            semi_supervision_keep=self.dataset.semi_variables["polar_semi_indexes"][k], epsilon=1e-6)
                    else:
                        self.polar_encoders[k].update_sliced_matrix(belief_mask, epsilon=1e-6)
                    polar_emb = self.polar_encoders[k].encode(polar_feature)
                    visualize_2d_tensor_and_save(polar_emb,
                                                 self.args.output_path / f"polar_emb_{epoch}_{k}.png")  ################## DEBUG
                    polar_A_pred = self.polar_encoders[k].decode(polar_emb)
                    pos_weights.append(self.pos_weight(self.polar_encoders[k].adj_sliced_target)[0])
                    polar_recon_loss = self.bce_loss_norm(
                        self.polar_encoders[k].adj_sliced_target) * F.binary_cross_entropy(
                        polar_A_pred.view(-1), self.polar_encoders[k].adj_sliced_target.reshape(-1),
                        weight=self.get_pos_weight_vector(self.polar_encoders[k].adj_sliced_target))
                    polar_recon_losses.append(polar_recon_loss)
                    if self.dataset.semi_variables["polar_semi_n"][k] > 0:
                        polar_semi_loss = self.compute_semi_loss(
                            polar_emb,
                            semi_adj_matrix=self.dataset.semi_variables["polar_semi_adj_matrix"][k],
                            semi_units=self.dataset.semi_variables["polar_semi_units"][k],
                            semi_indexes=self.dataset.semi_variables["polar_semi_indexes"][k],
                            semi_index_mapping=self.polar_encoders[k].to_slice_index
                        )
                        polar_semi_losses.append(polar_semi_loss)
                if polar_recon_losses:
                    loss += sum(polar_recon_losses) / len(polar_recon_losses)
                if polar_semi_losses:
                    loss += sum(polar_semi_losses) / len(polar_semi_losses)

            # Loss Function
            loss.backward()
            optimizer.step()

            # Logging
            if epoch % 1 == 0:
                polar_recon_losses_vis = [l.item() for l in polar_recon_losses]
                polar_semi_losses_vis = [l.item() for l in polar_semi_losses]
                print(f"[Epoch: {epoch} Total Loss: {loss.item():.4f}]")
                print(f"    Belief Rec Loss: {belief_recon_loss.item():.4f}")
                print(f"    Belief Semi Loss: {None if belief_semi_loss is None else belief_semi_loss.item():.4f}")
                if epoch >= self.args.belief_warmup:
                    print("    Polar Rec Loss:  {}  [{}]".format(
                        "{:.4f}".format(np.average(polar_recon_losses_vis)),
                        ", ".join([f"{l:.4f}" for l in polar_recon_losses_vis])))
                    print("    Polar Semi Loss: {}  [{}]".format(
                        None if not polar_semi_losses else "{:.4f}".format(np.average(polar_semi_losses_vis)),
                        " " if not polar_semi_losses else ", ".join([f"{l:.4f}" for l in polar_semi_losses_vis])))
                    print("    Polar Pos Weights: {}".format(", ".join([f"{int(l)}" for l in pos_weights])))
                    print("    Slice Results: User({}) -> [{}], Asser({}) -> [{}]".format(
                        self.polar_encoders[0].num_user,
                        ", ".join(
                            [str(self.polar_encoders[k].sliced_num_user) for k in range(len(self.polar_encoders))]),
                        self.polar_encoders[0].num_assertion,
                        ", ".join([str(self.polar_encoders[k].sliced_num_assertion) for k in
                                   range(len(self.polar_encoders))]),
                    ))

        # Save Training Results
        belief_emb = self.belief_encoder.encode(belief_feature)
        self.belief_embedding = belief_emb.cpu().detach().numpy()
        self.polar_embeddings = []
        for k in range(self.args.belief_dim):
            belief_mask = belief_emb[:, k]  # Belief Mask for Tweets [T,]
            if self.dataset.semi_variables["polar_semi_n"][k] > 0:
                self.polar_encoders[k].update_sliced_matrix(
                    belief_mask,
                    semi_supervision_keep=self.dataset.semi_variables["polar_semi_indexes"][k], epsilon=1e-6)
            else:
                self.polar_encoders[k].update_sliced_matrix(belief_mask, epsilon=1e-6)
            polar_emb = self.polar_encoders[k].encode(polar_feature)
            self.polar_embeddings.append(polar_emb.cpu().detach().numpy())
        print(self.belief_embedding)
        print(self.polar_embeddings)
