"""
Implementation of a Hidden Markov Model in PyTorch.
"""

import torch
import numpy as np


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

    def __init__(
        self,
        n_components,
        n_features,
        # startprob_=None,
        # transition=None,
        # emission=None,
        random_state=0,
        device='cpu',
        init_params='ste',
        params='ste',
        order=1,
        parallel=False,
        restarts=1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        # set seed with pytorch
        self.random_state = random_state
        self.device = device
        self.algorithm = 'viterbi'
        self.n_iter = 100,
        self.parallel = parallel
        self.order = order
        self.restarts = restarts
        torch.manual_seed(self.random_state)
        
        self.startprob_ = None
        self.transmat_ = None
        self.emissionprob_ = None

        # initialize parameters
        if 's' in init_params:
            startprob_ = torch.nn.Parameter(torch.rand(n_components), requires_grad=False)
            # if torch.any(startprob_ == 0):
            #     startprob_ = startprob_ + 1e-15
            #     startprob_ = startprob_ / startprob_.sum()        
            self.startprob_ = (startprob_ / startprob_.sum()).to(self.device)
        if 't' in init_params:
            transition = torch.nn.Parameter(torch.rand(n_components, n_components), requires_grad=False)
            self.transmat_ = (transition / transition.sum(axis=1)[:, None]).to(self.device)
        if 'e' in init_params:
            if parallel:
                emission = torch.nn.Parameter(torch.rand(restarts, n_components, n_features), requires_grad=False)
                self.emissionprob_ = (emission / emission.sum(axis=2)[:, :, None]).to(self.device)
            else:
                emission = torch.nn.Parameter(torch.rand(n_components, n_features), requires_grad=False)
                self.emissionprob_ = (emission / emission.sum(axis=1)[:, None]).to(self.device)
        
        # if any of the start_probs are equal to 0, add a small value (1e-15)
        # this is to avoid log(0) = -inf
        # start_prob = torch.where(start_prob == 0, torch.tensor(1e-15), start_prob)
        # If any values of start_prob == 0, apply smoothing such that no values end up being 0 but the sum of the probabilities is still 1
        # if torch.any(startprob_ == 0):
        #     startprob_ = startprob_ + 1e-15
        #     startprob_ = startprob_ / startprob_.sum()        

        # normalize parameters
        # self.startprob_ = (startprob_ / startprob_.sum()).to(self.device)
        # self.transmat_ = (transition / transition.sum(axis=1)[:, None]).to(self.device)
        # self.emissionprob_ = (emission / emission.sum(axis=1)[:, None]).to(self.device)

        # self.validate(self.startprob_, self.transmat_, self.emissionprob_)

    def validate(self, start_prob, transition, emission):
        if start_prob is not None and len(start_prob) != self.n_components:
            raise ValueError("start_prob must have length n_states")
        if transition is not None and transition.shape != (self.n_components, self.n_components):
            raise ValueError("transition must have shape (n_states, n_states)")
        if emission is not None and emission.shape != (self.n_components, self.n_features):
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
        alpha = torch.zeros(self.n_components, T).to(self.device)

        alpha[:, 0] = torch.log(self.startprob_) + \
            torch.log(self.emissionprob_[:, obs[0]].squeeze(1))
        
        for t in range(1, T):                
            alpha[:, t] = (
                torch.logsumexp(
                    alpha[:, t-1].unsqueeze(1) # [states, 1]
                    + torch.log(self.transmat_), # [states, states]
                    dim=0
                ) # [states, ]                
                + torch.log(self.emissionprob_[:, obs[t]].squeeze(1))) # [states, ]
                # slicing with a tensor obs[t] retains the original dimensionality

        return alpha


    def forward_algorithm_o2(self, obs):
        """Forward algoritm for second order HMM."""
        # startprob_ [n_components, ]
        # transmat_ [n_components, n_components, n_components]
        # emissionprob_ [n_components, n_features]
        T = len(obs)
        alpha = torch.zeros(self.n_components, self.n_components, T).to(self.device)
        alpha[:, :, 0] = (torch.log(self.startprob_) # [n_components, ]
                        + torch.log(self.emissionprob_[:, obs[0]].squeeze(1)) # [n_components, ]
                        ).unsqueeze(0) # [1, n_components]
        for t in range(1, T):
            alpha[:, :, t] = (
                torch.logsumexp(
                    alpha[:, :, t-1].unsqueeze(-1) # [n_components, n_components, 1]
                    + torch.log(self.transmat_), # [n_components, n_components, n_components]
                    dim=0 # sum over n_components
                ) # [n_components, n_components]
                + torch.log(self.emissionprob_[:, obs[t]].squeeze(-1).unsqueeze(0)) # [1, n_components]
            )
        
        return alpha

    
    def forward_algorithm_o2_p(self, obs):
        """Parallelized version of forward algorithm for second order HMM."""
        # startprob_ [n_components, ]
        # transmat_ [n_components, n_components, n_components]
        # emissionprob_ [restarts, n_components, n_features]
        # obs [T, ]
        T = len(obs)
        alpha = torch.zeros(self.restarts, self.n_components, self.n_components, T).to(self.device)
        alpha[:, :, :, 0] = (torch.log(self.startprob_) # [n_components, ]
                        + torch.log(self.emissionprob_[:, :, obs[0]].squeeze(2)) # [restarts, n_components]
                        ).unsqueeze(1) # [restarts, 1, n_components]
        for t in range(1, T):
            alpha[:, :, :, t] = (
                torch.logsumexp(
                    alpha[:, :, :, t-1].unsqueeze(-1) # [restarts, n_components, n_components, 1]
                  + torch.log(self.transmat_).expand(self.restarts, -1, -1, -1), # [restarts, n_components, n_components, n_components]
                    dim=1 # sum over n_components
                ) # [restarts, n_components, n_components]
                + torch.log(self.emissionprob_[:, :, obs[t]].squeeze(2).unsqueeze(1)) # [restarts, 1, n_components]
            )        
        return alpha


    def forward_algorithm_p(self, obs):
        """Parallelized version of forward algorithm"""                
        # startprob_ [n_components, ]
        # emissionprob_ [restarts, n_components, n_features]
        # transmat_ [n_components, n_components]
        # obs [T, ]
        
        T = len(obs)
        alpha = torch.zeros(self.restarts, self.n_components, T).to(self.device)
        alpha[:, :, 0] = (torch.log(self.startprob_) # [n_components, ]
                        + torch.log(self.emissionprob_[:, :, obs[0]].squeeze(2)) # [restarts, n_components]
        ) # [restarts, n_components]
        
        for t in range(1, T):                
            alpha[:, :, t] = (
                torch.logsumexp(
                    alpha[:, :, t-1].unsqueeze(-1) # [restarts, n_components, 1]
                    + torch.log(self.transmat_.expand(self.restarts, -1, -1)), # [restarts, n_components, n_components]
                    dim=1
                ) # [restarts, n_components]
                + torch.log(self.emissionprob_[:, :, obs[t]].squeeze(-1))) # [restarts, n_components]
                # slicing with a tensor obs[t] retains the original dimensionality

        return alpha # [restarts, n_components, T]


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
        if self.parallel:
            if self.order == 1:
                return self.score_p(obs)
            elif self.order == 2:
                return self.score_o2_p(obs)
        else:
            if self.order == 1:
                alpha = self.forward_algorithm(obs) # [states, T]
                score = torch.logsumexp(alpha[:, -1], dim=0)
                return score.item()
            elif self.order == 2:
                return self.score_o2(obs)

    
    def score_o2(self, obs):
        """Score for second order HMM"""
        alpha = self.forward_algorithm_o2(obs)
        score = torch.logsumexp(alpha[:, :, -1], dim=(0, 1))
        return score.item()

    
    def score_o2_p(self, obs):
        """Parallelized version of score function for second order HMM"""
        alpha = self.forward_algorithm_o2_p(obs)
        score = torch.logsumexp(alpha[:, :, :, -1], dim=(1, 2))
        return score # [restarts, ]

    
    def score_p(self, obs):
        """Parallelized version of score function"""
        alpha = self.forward_algorithm_p(obs)
        score = torch.logsumexp(alpha[:, :, -1], dim=1)
        return score # [restarts, ]


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
        if self.parallel:
            return self.backward_algorithm_p(obs)
        else:
            T = len(obs)
            beta = torch.zeros(self.n_components, T).to(self.device)
            # Last column can be left as all 0s instead of 1s because we are in log space
            beta[:, -1] = 1e-15
            
            for t in range(T-2, -1, -1):
                # print(f"{self.transition.shape}")
                # print(f"{torch.log(self.emission[:, obs[t+1]]).squeeze(1).unsqueeze(0).shape}")
                # print(f"{torch.log(beta[:, t+1]).unsqueeze(0).shape}")
                # foo
                # beta[:, t] = (self.transition @ (self.emission[:, obs[t+1]] * beta[:, t+1]))
                beta[:, t] = torch.logsumexp(
                    torch.log(self.transmat_) # [states, states]
                    + torch.log(self.emissionprob_[:, obs[t+1]]).squeeze(1).unsqueeze(0) # [1, states]
                    + beta[:, t+1].unsqueeze(0), # [1, states]
                    dim=1,
                )
                
            return beta

    
    def backward_algorithm_o2(self, obs):
        """Backward algorithm for second order HMM"""
        T = len(obs)
        beta = torch.zeros(self.n_components, self.n_components, T).to(self.device)
        # Last column can be left as all 0s instead of 1s because we are in log space
        beta[:, :, -1] = 1e-15
        
        for t in range(T-2, -1, -1):
            beta[:, :, t] = torch.logsumexp(
                torch.log(self.transmat_) # [n_components, n_components, n_components]
                + torch.log(self.emissionprob_[:, obs[t+1]]).squeeze(-1)[None, None, :] # [1, 1, n_components]
                + beta[:, :, t+1].unsqueeze(0), # [1, n_components, n_components]
                dim=2)

        return beta # [n_components, n_components, T]

    
    def backward_algorithm_o2_p(self, obs):
        """Parallelized version of backward algorithm for second order HMM"""
        T = len(obs)
        beta = torch.zeros(self.restarts, self.n_components, self.n_components, T).to(self.device)
        # Last column can be left as all 0s instead of 1s because we are in log space
        beta[:, :, :, -1] = 1e-15
        
        for t in range(T-2, -1, -1):
            beta[:, :, :, t] = torch.logsumexp(
                torch.log(self.transmat_.expand(self.restarts, -1, -1, -1)) # [restarts, n_components, n_components, n_components]
              + torch.log(self.emissionprob_[:, :, obs[t+1]]).squeeze(-1)[:, None, None, :] # [restarts, 1, 1, n_components]
              + beta[:, :, :, t+1].unsqueeze(1), # [restarts, 1, n_components, n_components],
                dim=3
            ) # [restarts, n_components, n_components]
        return beta # [restarts, n_components, n_components, T]

    
    def backward_algorithm_p(self, obs):
        """Parallelized version of backward algorithm"""
        T = len(obs)
        beta = torch.zeros(self.restarts, self.n_components, T).to(self.device)
        # Last column can be left as all 0s instead of 1s because we are in log space
        beta[:, :, -1] = 1e-15
        
        for t in range(T-2, -1, -1):
            beta[:, :, t] = torch.logsumexp(
                torch.log(self.transmat_.expand(self.restarts, -1, -1)) # [restarts, n_components, n_components]
                + torch.log(self.emissionprob_[:, :, obs[t+1]]).squeeze(-1).unsqueeze(1) # [restarts, 1, n_components]
                + beta[:, :, t+1].unsqueeze(1), # [restarts, 1, n_components]
                dim=2,
            )
        return beta # [restarts, n_components, T]


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
        gamma = torch.exp(
            gamma # [n_components, T]
            - torch.logsumexp(gamma, dim=0) # [T]
            ) # [n_components, T]
        
        # print(f"{gamma.shape=}")
        # print(f"{gamma.is_cuda=}")
        # print(f"{torch.nn.functional.one_hot(obs, self.n_obs).float().shape=}")

        self.emissionprob_ = gamma @ torch.nn.functional.one_hot(
            obs, self.n_features).float().squeeze(1)
        self.emissionprob_ /= self.emissionprob_.sum(axis=1)[:, None]

    
    def update_emission_o2(self, obs):
        """Update the emission matrix for second order HMM"""
        alpha = self.forward_algorithm_o2(obs)
        beta = self.backward_algorithm_o2(obs)
        gamma = alpha + beta
        gamma = torch.logsumexp(gamma, dim=0) # [n_components, T]
        gamma = torch.exp(
            gamma # [n_components, T]
            - torch.logsumexp(gamma, dim=0).unsqueeze(0) # [1, T]
            ) # [n_components, T]
        
        self.emissionprob_ = (gamma # [n_components, T]
        @ torch.nn.functional.one_hot(obs, self.n_features).float().squeeze(1) # [T, n_features]
        ) # [n_components, n_features]
        self.emissionprob_ /= self.emissionprob_.sum(axis=1)[:, None]

    
    def update_emission_o2_p(self, obs):
        """Parallelized version of update_emission_o2"""
        alpha = self.forward_algorithm_o2_p(obs)
        beta = self.backward_algorithm_o2_p(obs)
        gamma = alpha + beta # [restarts, n_components, n_components, T]
        gamma = torch.logsumexp(gamma, dim=1) # [restarts, n_components, T]
        gamma = torch.exp(
            gamma # [restarts, n_components, T]
          - torch.logsumexp(gamma, dim=1).unsqueeze(1) # [restarts, 1, T]
        ) # [restarts, n_components, T]
        self.emissionprob_ = (gamma # [restarts, n_components, T]
        @ torch.nn.functional.one_hot(obs, self.n_features).float().squeeze(1) # [T, n_features]
        ) # [restarts, n_components, n_features]
        self.emissionprob_ /= self.emissionprob_.sum(axis=2)[:, :, None]        


    def update_emission_p(self, obs):
        """Parallelized version of update_emission"""
        alpha = self.forward_algorithm_p(obs)
        beta = self.backward_algorithm_p(obs)
        gamma = alpha + beta
        gamma = torch.exp(
            gamma # [restarts, n_components, T]
            - torch.logsumexp(gamma, dim=1)[:, None, :] # [restarts, 1, T]
        ) # [restarts, n_components, T]        
        self.emissionprob_ = (
            gamma # [restarts, n_components, T]
            @ torch.nn.functional.one_hot(obs, self.n_features).float().squeeze(1) # [T, n_features]
            ) # [restarts, n_components, n_features]
        self.emissionprob_ /= self.emissionprob_.sum(axis=2)[:, :, None]


    def fit(self, X):
        tol = 1e-6
        # max_iter = 100000
        last_score = -np.inf
        
        for i in range(self.n_iter):
            if self.parallel:
                if self.order == 1:
                    self.update_emission_p(X)
                elif self.order == 2:
                    self.update_emission_o2_p(X)
                else:
                    raise ValueError("order must be one of 1 or 2")
            else:                
                if self.order == 1:
                    self.update_emission(X)
                elif self.order == 2:
                    self.update_emission_o2(X)
                    
            # check if score has converged
            score = self.score(X)
            # print(f"{i=}, {score=}")
            # if score < last_score:
            #     print("score decreased")
            #     break
            # if abs(last_score - score) < tol:
            #     break
            last_score = score
        # print(f"iter: {i}")


    def predict(self, X):
        if self.algorithm == "viterbi":
            return self.viterbi(X)
        elif self.algorithm == "map":
            if self.parallel:
                if self.order == 1:
                    return self.minimum_bayes_risk_p(X) 
                elif self.order == 2:
                    return self.minimum_bayes_risk_o2_p(X)
            else:
                if self.order == 1:
                    return self.minimum_bayes_risk(X)
                elif self.order == 2:
                    return self.minimum_bayes_risk_o2(X)
                else:
                    raise ValueError("order must be one of 1 or 2")
                
        else:
            raise ValueError("algorithm must be one of 'viterbi' or 'mbr'")


    def minimum_bayes_risk(self, obs):
        """
        Compute the minimum Bayes risk of the given sequence of observations.
        Parameters
        ----------
        obs : torch.Tensor
            Tensor of shape (n_obs,)
        Returns
        -------
        mbr : torch.Tensor
            Tensor of shape (n_obs,)
        """
        T = len(obs)
        alpha = self.forward_algorithm(obs)
        beta = self.backward_algorithm(obs)
        gamma = alpha + beta
        gamma = torch.exp(gamma - torch.logsumexp(gamma, dim=0))
        mbr = torch.argmax(gamma, dim=0)
        # return mbr as array
        mbr = mbr.cpu().numpy()
        return mbr

    
    def minimum_bayes_risk_o2(self, obs):
        T = len(obs)
        alpha = self.forward_algorithm_o2(obs) # [n_components, n_components, T]
        beta = self.backward_algorithm_o2(obs) # [n_components, n_components, T]
        gamma = torch.logsumexp(alpha + beta, dim=0) # [n_components, T]
        mbr = torch.argmax(gamma, dim=0)
        # return mbr as array
        mbr = mbr.cpu().numpy()
        return mbr

    
    def minimum_bayes_risk_o2_p(self, obs):
        T = len(obs)
        alpha = self.forward_algorithm_o2_p(obs) # [restarts, n_components, n_components, T]
        beta = self.backward_algorithm_o2_p(obs) # [restarts, n_components, n_components, T]
        gamma = torch.logsumexp(alpha + beta, dim=1) # [restarts, n_components, T]
        mbr = torch.argmax(gamma, dim=1) # [restarts, T]
        return mbr

    
    def minimum_bayes_risk_p(self, obs):
        """
        Compute the minimum Bayes risk of the given sequence of observations.
        """
        T = len(obs)
        alpha = self.forward_algorithm_p(obs)
        beta = self.backward_algorithm_p(obs)
        gamma = alpha + beta # [restarts, n_components, T]
        mbr = torch.argmax(gamma, dim=1) # [restarts, T]
        # return mbr as array
        # mbr = mbr.cpu().numpy()
        return mbr


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
        delta = torch.zeros(self.n_components, T)
        psi = torch.zeros(self.n_components, T, dtype=torch.long)
        delta[:, 0] = torch.log(self.startprob_) + \
            torch.log(self.emissionprob_[:, obs[0]])

        for t in range(1, T):
            delta[:, t] = torch.max(delta[:, t-1] + torch.log(self.transmat_), dim=1)[
                0] + torch.log(self.emissionprob_[:, obs[t]])
            psi[:, t] = torch.argmax(
                delta[:, t-1] + torch.log(self.transmat_), dim=1)

        path = torch.zeros(T, dtype=torch.long)
        path[T-1] = torch.argmax(delta[:, T-1])
        for t in range(T-2, -1, -1):
            path[t] = psi[path[t+1], t+1]
        return path


def main():    
    pass


if __name__ == "__main__":
    main()
