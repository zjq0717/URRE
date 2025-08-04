from module import *

class LatentModel(nn.Module):
    """
    Latent Model (Attentive Neural Process)
    """
    def __init__(self, num_hidden, input_dim):
        super(LatentModel, self).__init__()
        self.latent_encoder = LatentEncoder(num_hidden, num_hidden, input_dim)
        # self.deterministic_encoder = DeterministicEncoder(num_hidden, num_hidden, input_dim)
        self.decoder = Decoder(num_hidden, input_dim)
        self.BCELoss = nn.BCELoss()
        
    def forward(self, context_x, context_y, target_x, target_y=None):
        num_targets = target_x.size(1)

        prior_mu, prior_var, prior = self.latent_encoder(context_x, context_y)
        
        # For training
        if target_y is not None:
            posterior_mu, posterior_var, posterior = self.latent_encoder(target_x, target_y)
            z = posterior
        
        # For Generation
        else:
            z = prior

        z = z.unsqueeze(1).repeat(1,num_targets,1) # [B, T_target, H]

        # r = self.deterministic_encoder(context_x, context_y, target_x) # [B, T_target, H]

        # mu should be the prediction of target y
        y_pred = self.decoder(z, target_x)
        
        # For Training
        if target_y is not None:
            # get log probability
            bce_loss = self.BCELoss(t.sigmoid(y_pred), target_y)

            # get KL divergence between prior and posterior
            kl = self.kl_div(prior_mu, prior_var, posterior_mu, posterior_var)
            
            # maximize prob and minimize KL divergence
            loss = bce_loss + kl
        
        # For Generation
        else:
            log_p = None
            kl = None
            loss = None
        
        return y_pred, kl, loss, z
    
    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = (t.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / t.exp(prior_var) - 1. + (prior_var - posterior_var)
        kl_div = 0.5 * kl_div.sum()
        return kl_div


class LatentModel_new(nn.Module):
    # Vanilla Variational Auto-Encoder

    def __init__(self, input_x, input_y, latent_dim, hidden_dim=512, dropout=0.0, hidden_dim_r = 128):
        super(LatentModel_new, self).__init__()
        self.e1 = nn.Linear(input_x + input_y, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)
        self.r_dim = input_x
        self.e3 = nn.Linear(hidden_dim, self.r_dim)


        self.mean1 = nn.Linear(self.r_dim, hidden_dim_r)
        self.mean2 = nn.Linear(hidden_dim_r, latent_dim)
        self.log_std1 = nn.Linear(self.r_dim, hidden_dim_r)
        self.log_std2 = nn.Linear(hidden_dim_r, latent_dim)


        self.d1 = nn.Linear(latent_dim+input_x, hidden_dim_r)
        self.d2 = nn.Linear(hidden_dim_r, input_y)


        self.latent_dim = latent_dim
        self.device = device

    def forward(self, context_x, context_y, target_x, target_y=None):
        mean_C, std_C = self.encode(context_x, context_y)
        mean_T, std_T = self.encode(target_x, target_y)
        z_C_one = mean_C + std_C * torch.randn_like(std_C)
        batch_size = target_x.size(0)
        z_C = z_C_one.expand(batch_size, -1)

        u = self.decode(z_C, target_x)
        return u, mean_C, std_C, mean_T, std_T, z_C_one

    def encode(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], -1)))
        z = F.relu(self.e2(z))
        r = F.relu(self.e3(z))
        r = torch.mean(r, dim=0, keepdim=True)
        mean = self.mean1(r)
        mean = self.mean2(mean)
        log_std = self.log_std1(r)
        log_std = self.log_std2(log_std).clamp(-4, 15)
        std = torch.exp(log_std)
        return mean, std

    def decode(self, z, target_x):

        d = torch.cat([z, target_x], -1)
        a = F.relu(self.d1(d))
        return self.d2(a)

    def elbo_loss(self, state, action, beta=0.5):
        sa = torch.cat([state, action], -1)
        mean, std = self.encode(state, action)
        z = mean + std * torch.randn_like(std)
        u = self.decode(z)
        recon_loss = ((sa - u) ** 2).mean(-1)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1)
        vae_loss = recon_loss + beta * KL_loss
        return vae_loss