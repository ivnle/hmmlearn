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

    def __init__(self, n_states, n_obs, start_prob=None, transition=None, emission=None, random_state=0):
        super().__init__()
        self.n_states = n_states
        self.n_obs = n_obs
        # set seed with pytorch
        self.seed = random_state
        torch.manual_seed(self.seed)
        
        # initialize parameters
        start_prob = torch.nn.Parameter(torch.rand(
            n_states)) if start_prob is None else start_prob
        transition = torch.nn.Parameter(torch.rand(
            n_states, n_states)) if transition is None else transition
        emission = torch.nn.Parameter(torch.rand(
            n_states, n_obs)) if emission is None else emission
        
        # normalize parameters
        self.start_prob = start_prob / start_prob.sum()
        self.transition = transition / transition.sum(axis=1)[:, None]
        self.emission = emission / emission.sum(axis=1)[:, None]

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
        alpha = torch.zeros(self.n_states, T)
        # alpha[:, 0] = self.start_prob * self.emission[:, obs[0]]
        print(f"{obs[:10]=}")
        # print(f"{obs.shape=}")
        # print(f"{self.start_prob.shape=}")
        # print(f"{torch.log(self.emission[:, obs[0]]).squeeze(1).shape=}")

        print(f"{self.start_prob=}")
        print(f"{self.emission.sum(dim=1)=}")

        alpha[:, 0] = torch.log(self.start_prob) + \
            torch.log(self.emission[:, obs[0]].squeeze(1))

        print(f"{alpha=}")
        
        for t in range(1, T):
            # TODO torch.logsumexp(alpha[:, t-1] broadcasting? or be explicit?
            # alpha[:, t] = (alpha[:, t-1] @ self.transition) * self.emission[:, obs[t]]
            
            # print(f"{alpha[:, t-1].shape=}")
            # print(f"{torch.log(self.transition).shape=}")
            # print(f"{torch.log(self.emission[:, obs[t]]).shape=}")

            alpha[:, t] = (torch.logsumexp(alpha[:, t-1] + torch.log(self.transition), dim=1)
                           + torch.log(self.emission[:, obs[t]].squeeze(1)))
            print(f"{alpha=}")
        foo
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
        alpha = self.forward_algorithm(obs)
        return torch.logsumexp(alpha[:, -1], dim=0)

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
        beta = torch.zeros(self.n_states, T)
        beta[:, -1] = 1
        
        for t in range(T-2, -1, -1):
            # beta[:, t] = (self.transition @ (self.emission[:, obs[t+1]] * beta[:, t+1]))
            beta[:, t] = torch.logsumexp(
                torch.log(self.transition) + torch.log(self.emission[:, obs[t+1]]) + torch.log(beta[:, t+1]), dim=1
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
        gamma = alpha + beta
        gamma = torch.exp(gamma - torch.logsumexp(gamma, dim=0))
        
        # print(f"{gamma.shape=}")
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
