"""
Implementation of a Hidden Markov Model in PyTorch.
"""

import torch

class HMM(torch.nn.Module):
    """
    Hidden Markov Model.
    Parameters
    ----------
    n_states : int
        Number of hidden states.
    n_obs : int
        Number of observed states.
    start_prob : torch.Tensor, optional
        Initial state probabilities. If None, random values are used.
    """
    def __init__(self, n_Z, n_X, start_prob=None, transition=None, emission=None):
        super().__init__()
        self.n_Z = n_Z
        self.n_X = n_X

        self.validate(start_prob, transition, emission)
        
        # initialize parameters
        start_prob = torch.nn.Parameter(torch.rand(n_Z)) if start_prob is None else start_prob        
        transition = torch.nn.Parameter(torch.rand(n_Z, n_Z)) if transition is None else transition
        emission = torch.nn.Parameter(torch.rand(n_Z, n_X)) if emission is None else emission

        # normalize parameters
        self.start_prob = start_prob / start_prob.sum()
        self.transition = transition / transition.sum(axis=1)[:, None]
        self.emission = emission / emission.sum(axis=1)[:, None]

    def validate(self, start_prob, transition, emission):
        if start_prob is not None and len(start_prob) != self.n_Z:
            raise ValueError("start_prob must have length n_states")
        if transition is not None and transition.shape != (self.n_Z, self.n_Z):
            raise ValueError("transition must have shape (n_states, n_states)")            
        if emission is not None and emission.shape != (self.n_Z, self.n_X):
            raise ValueError("emission must have shape (n_states, n_obs)")
        return True

    def forward_algorithm(self, obs):
        """
        Compute the alpha matrix in a numerically stable fashion.
        Summing over the final column returns the probability
        of the given sequence of observations.
        
        Parameters
        ----------
        obs : torch.Tensor
            Tensor of shape (n_obs,)

        Returns
        -------
        alpha : torch.Tensor
            Tensor of shape (n_states, n_obs).
            alpha_ij = P(O_1, O_2, ..., O_j, q_j = S_i | theta)
        """
        alpha = torch.zeros(self.n_Z, self.n_X)
        # alpha[:, 0] = self.start_prob * self.emission[:, obs[0]]
        alpha[:, 0] = torch.log(self.start_prob) + torch.log(self.emission[:, obs[0]])

        T = len(obs)
        for t in range(1, T):
            # alpha[:, t] = (alpha[:, t-1] @ self.transition) * self.emission[:, obs[t]]
            alpha[:, t] = (torch.logsumexp(alpha[:, t-1] + torch.log(self.transition), dim=1)
                            + torch.log(self.emission[:, obs[t]]))
                    
        return alpha

    def score(self, obs):
        """
        Compute the probability of the given sequence of observations.
        Parameters
        ----------
        obs : torch.Tensor
            Tensor of shape (n_obs,)
        Returns
        -------
        score : float
            Probability of the given sequence of observations.
        """
        alpha = self.forward_algorithm(obs)
        return torch.logsumexp(alpha[:, -1], dim=0)
    

def main():
    model = HMM(n_Z=2, n_X=3)
    print(f"start_prob:\n {model.start_prob}")
    print(f"transition:\n {model.transition}")
    print(f"emission:\n {model.emission}")
    obs = torch.tensor([0, 1, 2, 1])
    alpha = model.forward_algorithm(obs)
    print(f"alpha:\n {alpha}")

if __name__ == "__main__":
    main()
