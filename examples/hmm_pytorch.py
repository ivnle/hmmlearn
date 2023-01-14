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

    def __init__(self, n_states, n_obs, start_prob=None, transition=None, emission=None, random_state=0, device='cpu'):
        super().__init__()
        self.n_states = n_states
        self.n_obs = n_obs
        # set seed with pytorch
        self.seed = random_state
        self.device = device
        torch.manual_seed(self.seed)
        
        # initialize parameters
        start_prob = torch.nn.Parameter(torch.rand(
            n_states)) if start_prob is None else start_prob
        transition = torch.nn.Parameter(torch.rand(
            n_states, n_states)) if transition is None else transition
        emission = torch.nn.Parameter(torch.rand(n_states, n_obs)) if emission is None else emission
        
        # if any of the start_probs are equal to 0, add a small value (1e-15)
        # this is to avoid log(0) = -inf
        # start_prob = torch.where(start_prob == 0, torch.tensor(1e-15), start_prob)
        # If any values of start_prob == 0, apply smoothing such that no values end up being 0 but the sum of the probabilities is still 1
        if torch.any(start_prob == 0):
            start_prob = start_prob + 1e-15
            start_prob = start_prob / start_prob.sum()        

        # normalize parameters
        self.start_prob = (start_prob / start_prob.sum()).to(self.device)
        self.transition = (transition / transition.sum(axis=1)[:, None]).to(self.device)
        self.emission = (emission / emission.sum(axis=1)[:, None]).to(self.device)

        self.validate(self.start_prob, self.transition, self.emission)

    def validate(self, start_prob, transition, emission):
        if start_prob is not None and len(start_prob) != self.n_states:
            raise ValueError("start_prob must have length n_states")
        if transition is not None and transition.shape != (self.n_states, self.n_states):
            raise ValueError("transition must have shape (n_states, n_states)")
        if emission is not None and emission.shape != (self.n_states, self.n_obs):
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
            alpha_ij = log P(O_1, O_2, ..., O_j, q_j = S_i | theta)
        """

        T = len(obs)
        alpha = torch.zeros(self.n_states, T).to(self.device)

        alpha[:, 0] = torch.log(self.start_prob) + \
            torch.log(self.emission[:, obs[0]].squeeze(1))
        
        for t in range(1, T):                
            alpha[:, t] = (
                torch.logsumexp(
                    alpha[:, t-1].unsqueeze(1) # [states, 1]
                    + torch.log(self.transition), # [states, states]
                    dim=0
                ) # [states, ]                
                + torch.log(self.emission[:, obs[t]].squeeze(1))) # [states, ]
                # slicing with a tensor obs[t] retains the original dimensionality

        return alpha

    def score_obs(self, obs):
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
        alpha = self.forward_algorithm(obs) # [states, T]
        score = torch.logsumexp(alpha[:, -1], dim=0)
        return score.item()

    def backward_algorithm(self, obs):
        """
        Compute the beta matrix in a numerically stable fashion.
        Parameters
        ----------
        obs : torch.Tensor
            Tensor of shape (n_obs,)
        Returns
        -------
        beta : torch.Tensor
            Tensor of shape (n_states, n_obs).
            beta_ij = log P(O_j+1, O_j+2, ..., O_T | q_j = S_i, theta)
        """
        T = len(obs)
        beta = torch.zeros(self.n_states, T).to(self.device)
        # Last column can be left as all 0s instead of 1s because we are in log space
        beta[:, -1] = 1e-15
        
        for t in range(T-2, -1, -1):
            # print(f"{self.transition.shape}")
            # print(f"{torch.log(self.emission[:, obs[t+1]]).squeeze(1).unsqueeze(0).shape}")
            # print(f"{torch.log(beta[:, t+1]).unsqueeze(0).shape}")
            # foo
            # beta[:, t] = (self.transition @ (self.emission[:, obs[t+1]] * beta[:, t+1]))
            beta[:, t] = torch.logsumexp(
                torch.log(self.transition) # [states, states]
                + torch.log(self.emission[:, obs[t+1]]).squeeze(1).unsqueeze(0) # [1, states]
                + beta[:, t+1].unsqueeze(0), # [1, states]
                dim=1,
            )
            
        return beta

    def update_emission(self, obs):
        """
        Update the emission matrix.
        Parameters
        ----------
        obs : torch.Tensor
            Tensor of shape (n_obs,)
        """
        alpha = self.forward_algorithm(obs)
        beta = self.backward_algorithm(obs)

        # print(f"{alpha=}")
        # print(f"{beta=}")
        # foo

        gamma = alpha + beta
        gamma = torch.exp(gamma - torch.logsumexp(gamma, dim=0)) # [states, T]
        
        # print(f"{gamma.shape=}")
        # print(f"{gamma.is_cuda=}")
        # print(f"{torch.nn.functional.one_hot(obs, self.n_obs).float().shape=}")

        self.emission = gamma @ torch.nn.functional.one_hot(
            obs, self.n_obs).float().squeeze(1)
        self.emission /= self.emission.sum(axis=1)[:, None]

    def viterbi(self, obs):
        """
        Compute the most likely sequence of hidden states.
        Parameters
        ----------
        obs : torch.Tensor
            Tensor of shape (n_obs,)
        Returns
        -------
        path : torch.Tensor
            Tensor of shape (n_obs,)
        """
        T = len(obs)
        delta = torch.zeros(self.n_states, T)
        psi = torch.zeros(self.n_states, T, dtype=torch.long)
        delta[:, 0] = torch.log(self.start_prob) + \
            torch.log(self.emission[:, obs[0]])

        for t in range(1, T):
            delta[:, t] = torch.max(delta[:, t-1] + torch.log(self.transition), dim=1)[
                0] + torch.log(self.emission[:, obs[t]])
            psi[:, t] = torch.argmax(
                delta[:, t-1] + torch.log(self.transition), dim=1)

        path = torch.zeros(T, dtype=torch.long)
        path[T-1] = torch.argmax(delta[:, T-1])
        for t in range(T-2, -1, -1):
            path[t] = psi[path[t+1], t+1]
        return path


def main():    
    pass


if __name__ == "__main__":
    main()
