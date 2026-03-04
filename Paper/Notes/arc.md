以下は提供された論文テキストをMarkdown形式で書き起こしたものです。数式はLaTeX形式で記述し、図表やアルゴリズムは読みやすい形式に整えています。

---

# ARC - Actor Residual Critic for Adversarial Imitation Learning

**Ankur Deka†∗, Changliu Liu†, Katia Sycara†**
†Robotics Institute, Carnegie Mellon University
∗Intel Labs
`adeka@alumni.cmu.edu`, `{cliu6,katia}@cs.cmu.edu`

6th Conference on Robot Learning (CoRL 2022), Auckland, New Zealand.
arXiv:2206.02095v4 [cs.LG] 30 Nov 2022

---

## Abstract

Adversarial Imitation Learning (AIL) is a class of popular state-of-the-art Imitation Learning algorithms commonly used in robotics. In AIL, an artificial adversary’s misclassification is used as a reward signal that is optimized by any standard Reinforcement Learning (RL) algorithm. Unlike most RL settings, the reward in AIL is differentiable but current model-free RL algorithms do not make use of this property to train a policy. The reward is AIL is also shaped since it comes from an adversary. We leverage the differentiability property of the shaped AIL reward function and formulate a class of Actor Residual Critic (ARC) RL algorithms. ARC algorithms draw a parallel to the standard Actor-Critic (AC) algorithms in RL literature and uses a residual critic, C function (instead of the standard Q function) to approximate only the discounted future return (excluding the immediate reward). ARC algorithms have similar convergence properties as the standard AC algorithms with the additional advantage that the gradient through the immediate reward is exact. For the discrete (tabular) case with finite states, actions, and known dynamics, we prove that policy iteration with C function converges to an optimal policy. In the continuous case with function approximation and unknown dynamics, we experimentally show that ARC aided AIL outperforms standard AIL in simulated continuous-control and real robotic manipulation tasks. ARC algorithms are simple to implement and can be incorporated into any existing AIL implementation with an AC algorithm. Video and link to code are available at: [sites.google.com/view/actor-residual-critic](https://sites.google.com/view/actor-residual-critic).

**Keywords:** Adversarial Imitation Learning (AIL), Actor-Critic (AC), Actor Residual Critic (ARC)

---

## 1. Introduction

Although Reinforcement Learning (RL) allows us to train agents to perform complex tasks without manually designing controllers [1, 2, 3], it is often tedious to hand-craft a dense reward function that captures the task objective in robotic tasks [4, 5, 6]. Imitation Learning (IL) or Learning from Demonstration (LfD) is a popular choice in such situations [4, 5, 6, 7]. Common approaches to IL are Behavior Cloning (BC) [8] and Inverse Reinforcement Learning (IRL) [9].

Within IRL, recent Adversarial Imitation Learning (AIL) algorithms have shown state-of-the-art performance, especially in continuous control tasks which make them relevant to real-world robotics problems. AIL methods cast the IL problem as an adversarial game between a policy and a learned adversary (discriminator). The adversary aims to classify between agent and expert trajectories and the policy is trained using the adversary’s mis-classification as the reward function. This encourages the policy to imitate the expert. Popular AIL algorithms include Generative Adversarial Imitation Learning (GAIL) [10], Adversarial Inverse Reinforcement Learning (AIRL) [11] and f-MAX [12].

The agent in AIL is trained with any standard RL algorithm. There are two popular categories of RL algorithms: (i) on-policy algorithms such as TRPO [13], PPO [2], GAE [14] based on the policy gradient theorem [15, 16]; and (ii) off-policy Actor-Critic (AC) algorithms such as DDPG [17], TD3 [18], SAC [3] that compute the policy gradient through a critic (Q function). These standard RL algorithms were designed for arbitrary scalar reward functions; and they compute an approximate gradient for updating the policy. Practical on-policy algorithms based on the policy gradient theorem use several approximations to the true gradient [13, 2, 14] and off-policy AC algorithms first approximate policy return with a critic (Q function) and subsequently compute the gradient through this critic [17, 18, 3]. Even if the Q function is approximated very accurately, the error in its gradient can be arbitrarily large, Appendix A.1.

Our insight is that the reward function in AIL has 2 special properties: (i) it is differentiable which means we can compute the exact gradient through the reward function instead of approximating it and (ii) it is dense/shaped as it comes from an adversary. As we will see in section 3, naively computing the gradient through reward function would lead to a short-sighted sub-optimal policy. To address this issue, we formulate a class of Actor Residual Critic (ARC) RL algorithms that use a residual critic, C function (instead of the standard Q function) to approximate only the discounted future return (excluding immediate reward).

The contribution of this paper is the introduction of ARC, which can be easily incorporated to replace the AC algorithm in any existing AIL algorithm for continuous-control and helps boost the asymptotic performance by computing the exact gradient through the shaped reward function.

---

## 2. Related Work

**Table 1: Popular AIL algorithms, f-divergence metrics they minimize and their reward functions.**

| Algorithm Name | Minimized f-Divergence | Reward $r(s, a)$ Expression |
| :--- | :--- | :--- |
| GAIL [10] | Jensen-Shannon | $-\log(1 - D(s, a))$ (derived) |
| AIRL [11], f-MAX-RKL [12] | Reverse KL | $\log D(s, a) - \log(1 - D(s, a))$ |

*Note: The table content in the raw text was fragmented. The above represents the standard mapping described in the cited papers and text context.*

The simplest approach to imitation learning is Behavior Cloning [8] where an agent policy directly regresses on expert actions (but not states) using supervised learning. This leads to distribution shift and poor performance at test time [19, 10]. Methods such as DAgger [19] and Dart [20] eliminate this issue but assume an interactive access to an expert policy, which is often impractical.

Inverse Reinforcement Learning (IRL) approaches recover a reward function which can be used to train an agent using RL [9, 21] and have been more successful than BC. Within IRL, recent Adversarial Imitation Learning (AIL) methods inspired by Generative Adversarial Networks (GANs) [22] have been extremely successful. GAIL [10] showed state-of-the-art results in imitation learning tasks following which several extensions have been proposed [23, 24]. AIRL [11] imitates an expert as well as recovers a robust reward function. [25] and [12] presented a unifying view on AIL methods by showing that they minimize different divergence metrics between expert and agent state-action distributions but are otherwise similar. [12] also presented a generalized AIL method f-MAX which can minimize any specified f-divergence metric [26] between expert and agent state-action distributions thereby imitating the expert. Choosing different divergence metrics leads to different AIL algorithms, e.g. choosing Jensen-Shannon divergence leads to GAIL [10]. [27] proposed a method that automatically learns a f-divergence metric to minimize. Our proposed Actor Residual Critic (ARC) can be augmented with any of these AIL algorithms to leverage the reward gradient.

Some recent methods have leveraged the differentiable property of reward in certain scenarios but they have used this property in very different settings. [28] used the gradient of the reward to improve the reward function but not to optimize the policy. We on the other hand explicitly use the gradient of the reward to optimize the policy. [29] used the gradient through the reward to optimize the policy but operated in the model-based setting. If we have access to a differentiable dynamics model, we can directly obtain the gradient of the expected return (policy objective) w.r.t. the policy parameters, Appendix E.5. Since we can directly obtain the objective’s gradient, we do not necessarily need to use either a critic (Q) as in standard Actor Critic (AC) algorithms or a residual critic (C) as in our proposed Actor Residual Critic (ARC) algorithms. Differentiable cost (negative reward) has also been leveraged in control literature for a long time to compute a policy, e.g. in LQR [30] and its extensions; but they assume access to a known dynamics model. We on the other hand present a model-free method with unknown dynamics that uses the gradient of the reward to optimize the policy with the help of a new class of RL algorithms called Actor Residual Critic (ARC).

---

## 3. Background

**Objective** Our goal is to imitate an expert from one or more demonstrated trajectories (state-action sequences) in a continuous-control task (state and action spaces are continuous). Given any Adversarial Imitation Learning (AIL) algorithm that uses an off-policy Actor-Critic algorithm RL algorithm, we wish to use our insight on the availability of a differentiable reward function to improve the imitation learning algorithm.

**Notation** The environment is modeled as a Markov Decision Process (MDP) represented as a tuple $(S, A, P, r, \rho_0, \gamma)$ with state space $S$, action space $A$, transition dynamics $P : S \times A \times S \to [0, 1]$, reward function $r(s, a)$, initial state distribution $\rho_0(s)$, and discount factor $\gamma$. $\pi(.|s)$, $\pi_{exp}(.|s)$ denote policies and $\rho^\pi, \rho_{exp} : S \times A \to [0, 1]$ denote state-action occupancy distributions for agent and expert respectively. $T = \{s_1, a_1, s_2, a_2, . . . , s_T, a_T\}$ denotes a trajectory or episode and $(s, a, s', a')$ denotes a continuous segment in a trajectory. A discriminator or adversary $D(s, a)$ tries to determine whether the particular $(s, a)$ pair belongs to an expert trajectory or agent trajectory, i.e. $D(s, a) = P(\text{expert}|s, a)$. The optimal discriminator is $D(s, a) = \frac{\rho_{exp}(s,a)}{\rho_{exp}(s,a)+\rho_\pi(s,a)}$ [22].

**Adversarial Imitation Learning (AIL)** In AIL, the discriminator and agent are alternately trained. The discriminator is trained to maximize the likelihood of correctly classifying expert and agent data using supervised learning, (1) and the agent is trained to maximize the expected discounted return, (2).

$$ \max_{D} \left\{ \mathbb{E}_{s,a \sim \rho_{exp}} [\log D(s, a)] + \mathbb{E}_{s,a \sim \rho_\pi} [\log(1 - D(s, a))] \right\} \quad (1) $$

$$ \max_{\pi} \left\{ \mathbb{E}_{s,a \sim \rho_{0,\pi,P}} \sum_{t \ge 0} \gamma^t r(s_t, a_t) \right\} \quad (2) $$

Here, reward $r_\psi(s, a) = h(D_\psi(s, a))$ is a function of the discriminator which varies between different AIL algorithms. Different AIL algorithms minimize different f-divergence metrics between expert and agent state-action distribution. Defining a f-divergence metric instantiates different reward functions [12]. Some popular divergence choices are Jensen-Shannon in GAIL [10] and Reverse Kullback-Leibler in f-MAX-RKL [12] and AIRL [11] as shown in Table 1.

Any RL algorithm could be used to optimize (2) and popular choices are off-policy Actor-Critic algorithms such as DDPG [17], TD3 [18], SAC [3] and on-policy algorithms such as TRPO [13], PPO [2], GAE [14] which are based on the policy gradient theorem [15, 16]. We focus on off-policy Actor-Critic algorithms as they are usually more sample efficient and stable than on-policy policy gradient algorithms [18, 3].

**Continuous-control using off-policy Actor-Critic** The objective in off-policy RL algorithms is to maximize expected Q function of the policy, $Q^\pi$ averaged over the state distribution of a dataset $\mathcal{D}$ (typically past states stored in buffer) and the action distribution of the policy $\pi$ [31]:

$$ \max_{\pi} \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi} Q^\pi(s, a) \quad (3) $$

where,
$$ Q^\pi(s, a) = \mathbb{E}_{s,a \sim \rho_{0,\pi,P}} \left[ \sum_{k \ge 0} \gamma^k r_{t+k} \mid s_t = s, a_t = a \right] \quad (4) $$

The critic and the policy denoted by $Q, \pi$ respectively are approximated by function approximators such as neural networks with parameters $\phi$ and $\theta$ respectively. There is an additional target $Q_{\phi_{targ}}$ function parameterized by $\phi_{targ}$. There are two alternating optimization steps:

1.  **Policy evaluation:** Fit critic ($Q_\phi$ function) by minimizing Bellman Backup error.
    $$ \min_{\phi} \mathbb{E}_{s,a,s' \sim \mathcal{D}} \{Q_\phi(s, a) - y(s, a)\}^2 \quad (5) $$
    where, $y(s, a) = r(s, a) + \gamma Q_{\phi_{targ}} (s', a') \text{ and } a' \sim \pi_\theta(.|s') \quad (6)$
    $Q_\phi$ is updated with gradient descent without passing gradient through the target $y(s, a)$.

2.  **Policy improvement:** Update policy with gradient ascent over RL objective.
    $$ \mathbb{E}_{s \sim \mathcal{D}} \nabla_\theta Q_\phi(s, a \sim \pi_\theta(.|s)) \quad (7) $$

All off-policy Actor Critic algorithms follow the core idea above ((5) and (7)) along with additional details such as the use of a deterministic policy and target network in DDPG [17], double Q networks and delayed updates in TD3 [18], entropy regularization and reparameterization trick in SAC [3].

**Naive-Diff and why it won’t work** Realizing that the reward in AIL is differentiable and shaped, we can formulate a Naive-Diff RL algorithm that updates the policy by differentiating the RL objective (2) with respect to the policy parameters $\theta$.

$$ \mathbb{E}_{T \sim \mathcal{D}} \nabla_\theta r(s_1, a_1) + \gamma \nabla_\theta r(s_2, a_2) + \gamma^2 \nabla_\theta r(s_3, a_3) + . . . \quad (8) $$

$T = \{s_1, a_1, s_2, a_2 . . . \}$ is a sampled trajectory in $\mathcal{D}$. Using standard autodiff packages such as Pytorch [32] or Tensorflow [33] to naively compute the gradients in (8) would produce incorrect gradients. Apart from the immediate reward $r(s_1, a_1)$, all the terms depend on the transition dynamics of the environment $P(s_{t+1}|s_t, a_t)$, which is unknown and we cannot differentiate through it. So, autodiff will calculate the gradient of only immediate reward correctly and calculate the rest as 0’s. This will produce a short-sighted sub-optimal policy that maximizes only the immediate reward.

---

## 4. Method

*(Figure 1 description: Visual illustration of approximating reward via Q function or C function. Q approximates return ($r_1 + r_2 + r_3 + \dots$). C approximates future return (residue) ($r_2 + r_3 + \dots$), while Immediate reward is kept separate.)*

The main lesson we learnt from Naive-Diff is that while we can obtain the gradient of immediate reward, we cannot directly obtain the gradient of future return due to unknown environment dynamics. This directly motivates our formulation of Actor Residual Critic (ARC). Standard Actor Critic algorithms use Q function to approximate the return as described in Eq. 4. However, since we can directly obtain the gradient of the reward, we needn’t approximate it with a Q function. We, therefore, propose to use C function to approximate only the future return, leaving out the immediate reward. This is the core idea behind Actor Residual Critic (ARC) and is highlighted in Fig. 1. The word “Residual” refers to the amount of return that remains after subtracting the immediate reward from the return. As we will see in Section 4.3, segregating the immediate reward from future return will allow ARC algorithms to leverage the exact gradient of the shaped reward. We now formally describe Residual Critic (C function) and its relation to the standard critic (Q function).

### 4.1 Definition of Residual Critic (C function)

The Q function under a policy $\pi$, $Q^\pi(s, a)$, is defined as the expected discounted return from state $s$ taking action $a$, (9). The C function under a policy $\pi$, $C^\pi(s, a)$, is defined as the expected discounted future return, excluding the immediate reward (10). Note that the summation in (10) starts from 1 instead of 0. Q function can be expressed in terms of C function as shown in (11).

$$ Q^\pi(s, a) = \mathbb{E}_{s,a \sim \rho_{0,\pi,P}} \left[ \sum_{k \ge 0} \gamma^k r_{t+k} \mid s_t = s, a_t = a \right] \quad (9) $$

$$ C^\pi(s, a) = \mathbb{E}_{s,a \sim \rho_{0,\pi,P}} \left[ \sum_{k \ge 1} \gamma^k r_{t+k} \mid s_t = s, a_t = a \right] \quad (10) $$

$$ Q^\pi(s, a) = r(s, a) + C^\pi(s, a) \quad (11) $$

### 4.2 Policy Iteration using C function

Using C function, we can formulate a Policy Iteration algorithm as shown in Algorithm 1, which is guaranteed to converge to an optimal policy (Theorem 1), similar to the case of Policy Iteration with Q or V function [16]. Other properties of C function and proofs are presented in Appendix B.

**Algorithm 1: Policy Iteration with C function**
```
Initialize C_0(s, a) for all s, a;
while π not converged do
    // Policy evaluation
    for n=1,2,... until C_k converges do
        C_{n+1}(s, a) ← γ Σ_{s'} P(s'|s, a) Σ_{a'} π(a'|s') (r(s', a') + C_n(s', a'))  for all s, a
    
    // Policy improvement
    π(s, a) ← { 1, if a = argmax_{a'} (r(s, a') + C(s, a'))
              { 0, otherwise
              for all s, a
```

### 4.3 Continuous-control using Actor Residual Critic

We can easily extend the policy iteration algorithm with C function (Algorithm 1) for continuous-control tasks using function approximators instead of discrete C values and a discrete policy (similar to the case of Q function [16]). We call any RL algorithm that uses a policy, $\pi$ and a residual critic, $C$ function as an Actor Residual Critic (ARC) algorithm. Using the specific details of different existing Actor Critic algorithms, we can formulate analogous ARC algorithms. For example, using a deterministic policy and target network as in [17] we can get ARC-DDPG. Using double C networks (instead of Q networks) and delayed updates as in [18] we can get ARC-TD3. Using entropy regularization and reparameterization trick as in [3] we can get ARC-SAC or SARC (Soft Actor Residual Critic).

### 4.4 ARC aided Adversarial Imitation Learning

To incorporate ARC in any Adversarial Imitation Learning algorithm, we simply replace the Actor Critic RL algorithm with an ARC RL algorithm without altering anything else in the pipeline. For example, we can replace SAC [3] with SARC to get SARC-AIL as shown in Algorithm 2. Implementation-wise this is extremely simple and doesn’t require any additional functional parts in the algorithm. The same neural network that approximated Q function can be now be used to approximate C function.

**Algorithm 2: SARC-AIL: Soft Actor Residual Critic Adversarial Imitation Learning**
```
Intialization: Environment (env), Discriminator parameters ψ, Policy parameters θ, 
               C-function parameters φ1, φ2, dataset of expert demonstrations D_exp, 
               replay buffer D, Target parameters φtarg1 ← φ1, φtarg2 ← φ2, 
               Entropy regularization coefficient α;

while Max no. of environment interactions is not reached do
    a ∼ πθ(.|s);
    s', r, d = env.step(a); d = 1 if s' is terminal state, 0 otherwise
    Store (s, a, s', d) in replay buffer D;
    
    if Update interval reached then
        for no. of update steps do
            Sample batch B = (s, a, s', d) ∼ D;
            Sample batch of expert demonstrations B_exp = (s, a) ∼ Dexp;
            
            Update Discriminator parameters (ψ) with gradient ascent.
            ∇ψ { Σ_{(s,a)∈Bexp} [log Dψ(s, a)] + Σ_{(s,a,s',d)∈B} [log(1 − Dψ(s, a))] };
            
            Compute C targets ∀(s, a, s', d) ∈ B
            y(s, a, d) = γ [ rψ(s', ã') + min_{i=1,2} Cφtargi(s', ã') − α log πθ(ã'|s') ]
            where ã' ∼ πθ(.|s'), rψ(s', ã') = h(Dψ(s', ã'))
            
            Update C-functions parameters (φ1, φ2) with gradient descent.
            ∇φi (1/|B|) Σ_{(s,a,s',d)∈B} (Cφi(s, a) − y(s, a, d))², for i = 1, 2
            
            Update policy parameters (θ) with gradient ascent.
            ∇θ (1/|B|) Σ_{s∈B} [ rψ(s, ã) + min_{i=1,2} Cφi(s, ã) − α log πθ(ã|s) ]
            where ã ∼ πθ(.|s), rψ(s, ã) = h(Dψ(s, ã))
            
            Update target networks.
            φtargi ← ζφtargi + (1 − ζ)φi, for i = 1, 2; ζ controls polyak averaging
```

### 4.5 Why choose ARC over Actor-Critic in Adversarial Imitation Learning?

The advantage of using an ARC algorithm over an Actor-Critic (AC) algorithm is that we can leverage the exact gradient of the reward. Standard AC algorithms use $Q_\phi$ to approximate the immediate reward + future return and then compute the gradient of the policy parameters through the $Q_\phi$ function (12). This is an approximate gradient with no bound on the error in gradient, since the $Q_\phi$ function is an estimated value, Appendix A.1. On the other hand, ARC algorithms segregate the immediate reward (which is known in Adversarial Imitation Learning) from the future return (which needs to be estimated). ARC algorithms then compute the gradient of policy parameters through the immediate reward (which is exact) and the C function (which is approximate) separately (13).

Standard AC:
$$ \mathbb{E}_{s \sim \mathcal{D}} [\nabla_\theta Q_\phi(s, a)], a \sim \pi_\theta(.|s) \quad (12) $$

ARC (Our):
$$ \mathbb{E}_{s \sim \mathcal{D}} [\nabla_\theta r(s, a) + \nabla_\theta C_\phi(s, a)], a \sim \pi_\theta(.|s) \quad (13) $$

In Appendix A.2, we derive the conditions under which ARC is likely to outperform AC by performing a (Signal to Noise Ratio) SNR analysis similar to [34]. Intuitively, favourable conditions for ARC are (i) Error in gradient due to function approximation being similar or smaller for C as compared to Q (ii) the gradient of the immediate reward not having a high negative correlation with the gradient of C ($\mathbb{E} [\nabla_a r(s, a) \nabla_a C(s, a)]$ is not highly negative). Under these conditions, ARC would produce a higher SNR estimate of the gradient to train the policy. We believe that AIL is likely to present favourable conditions for ARC since the reward is shaped.

ARC would under-perform AC if the error in gradient due to function approximation of C network is significantly higher than that of Q network. In the general RL setting, immediate reward might be misleading (i.e. $\mathbb{E} [\nabla_a r(s, a) \nabla_a C(s, a)]$ might be negative) which might hurt the performance of ARC. However, we propose using ARC for AIL where the adversary reward measures how closely the agent imitates the expert. In AIL, the adversary reward is dense/shaped making ARC likely to be useful in this scenario, as experimentally verified in the following section.

---

## 5. Results

In Theorem 1, we proved that Policy Iteration with C function converges to an optimal policy. In Fig. 2, we experimentally validate this on an example grid world. The complete details are presented in Appendix E.1. In the following sections (5.2, 5.3 and 5.4) we show the effectiveness of ARC aided AIL in Mujoco continuous-control tasks, and simulated and real robotic manipulation tasks. In Appendix D.2, we experimentally illustrate that ARC produces more accurate gradients than AC using a simple 1D driving environment. The results are discussed in more detail in Appendix F.

### 5.1 Policy Iteration on a Grid World

*(Figure 2 description: Results on a Grid World comparing Policy Iteration (PI) with C function vs PI with Q function. Both converge in 7 steps to the same optimal policy $\pi^*$. The figure shows visual maps of $\pi^*$, $r^*$, $C^*$, and $Q^*$, verifying the relation $Q^* = r^* + C^*$.)*

### 5.2 Imitation Learning in Mujoco continuous-control tasks

We used 4 Mujoco continuous-control environments from OpenAI Gym [35], as shown in Fig. 3 (Ant, Walker, HalfCheetah, Hopper). Expert trajectories were obtained by training a policy with SAC [3]. We evaluated the benefit of using ARC with two popular Adversarial Imitation Learning (AIL) algorithms, f-MAX-RKL [12] and GAIL [10]. For each of these algorithms, we evaluated the performance of standard AIL algorithms (f-MAX-RKL, GAIL), ARC aided AIL algorithms (ARC-f-MAX-RKL, ARC-GAIL) and Naive-Diff algorithm described in Section 3 (Naive-Diff-f-MAX-RKL, Naive-Diff-GAIL). We also evaluated the performance of Behavior Cloning (BC). For standard AIL algorithms (GAIL and f-MAX-RKL) and BC, we used the implementation of [28]. Further experimental details are presented in Appendix E.

**Table 2: Policy return on Mujoco environments using different Imitation Learning algorithms.**
*(Each algorithm is run with 10 random seeds. Each seed is evaluated for 20 episodes.)*

| Method | Ant | Walker2d | HalfCheetah | Hopper |
| :--- | :--- | :--- | :--- | :--- |
| Expert return | 5926.18 ± 124.56 | 5344.21 ± 84.45 | 12427.49 ± 486.38 | 3592.63 ± 19.21 |
| **ARC-f-Max-RKL (Our)** | **6306.25 ± 95.91** | **4753.63 ± 88.89** | **12930.51 ± 340.02** | 3433.45 ± 49.48 |
| f-Max-RKL | 5949.81 ± 98.75 | 4069.14 ± 52.14 | 11970.47 ± 145.65 | 3417.29 ± 19.8 |
| Naive-Diff f-Max-RKL | 998.27 ± 3.63 | 294.36 ± 31.38 | 357.05 ± 732.39 | 154.57 ± 34.7 |
| **ARC-GAIL (Our)** | 6090.19 ± 99.72 | 3971.25 ± 70.11 | 11527.76 ± 537.13 | 3392.45 ± 10.32 |
| GAIL | 5907.98 ± 44.12 | 3373.26 ± 98.18 | 11075.31 ± 255.69 | 3153.84 ± 53.61 |
| Naive-Diff GAIL | 998.17 ± 2.22 | 99.26 ± 76.11 | 277.12 ± 523.77 | 105.3 ± 48.01 |
| BC | 615.71 ± 109.9 | 81.04 ± 119.68 | -392.78 ± 74.12 | 282.44 ± 110.7 |

*(Figure 4 description: Episode return versus number of environment interaction steps plots. ARC methods show consistently better performance than standard AIL methods across Ant, Walker2d, HalfCheetah, and Hopper.)*

### 5.3 Imitation Learning in robotic manipulation tasks

We used simplified 2D versions of FetchReach (Fig. 5a) and FetchPush (Fig. 5b) robotic manipulation tasks from OpenAI Gym [35] which have a simulated Fetch robot, [36]. In the FetchReach task, the robot needs to take it’s end-effector to the goal (virtual red sphere) as quickly as possible. In the FetchPush task, the robot’s needs to push the block to the goal as quickly as possible. We used hand-coded proportional controller to generate expert trajectories for these tasks. Further details are presented in Appendix E.3.

Fig. 6a shows the training plots and Table 3 under the heading ‘Simulation’ shows the final performance of the different algorithms. In both the FetchReach and FetchPush tasks, ARC aided AIL algorithms consistently outperformed the standard AIL algorithms. Fig. 6b shows the magnitude of the 2nd action dimension vs. time-step in one episode for different algorithms. The expert initially executed large actions when the end-effector/block was far away from the goal. As the end-effector/block approached the goal, the expert executed small actions. ARC aided AIL algorithms (ARC-f-Max-RKL and ARC-GAIL) showed a similar trend while standard AIL algorithms (f-Max-RKL and GAIL) learnt a nearly constant action. Thus, ARC aided AIL algorithms were able to better imitate the expert than standard AIL algorithms.

### 5.4 Sim-to-real transfer of robotic manipulation policies

For testing the sim-to-real transfer of the different trained AIL manipulation policies, we setup JacoReach (Fig. 5c) and JacoPush (Fig. 5d) tasks with a Kinova Jaco Gen 2 arm, similar to the FetchReach and FetchPush tasks in the previous section. The details are presented in Appendix E.4. Table 3 under the heading ‘Real Robot’ shows the performance of the different AIL algorithms in the real robotic manipulation tasks. The real robot evaluations showed a similar trend as in the simulated tasks. ARC aided AIL consistently outperformed the standard AIL algorithms. Appendix D Fig. 9 visualizes the policies in the JacoPush task showing that ARC aided AIL algorithms were able to push the block closer to the goal as compared to the standard AIL algorithms. Project website contains videos of the same. Since we didn’t tune hyper-parameters for these tasks (both our methods and the baselines, details in Appendix E.3), it is likely that the performances would improve with further parameter tuning. Without fine-tuning hyper-parameters for these tasks, ARC algorithms showed higher performance than the baselines. This shows that ARC algorithms are parameter robust and applicable to real robot tasks without much fine tuning.

**Table 3: Policy return on simulated (FetchReach, FetchPush) and real (JacoReach, JacoPush) robotic manipulation tasks using different AIL algorithms.**

| | Simulation | | Real Robot | |
| :--- | :--- | :--- | :--- | :--- |
| **Method** | **FetchReach** | **FetchPush** | **JacoReach** | **JacoPush** |
| Expert return | -0.58 ± 0 | -1.18 ± 0.04 | -0.14 ± 0.01 | -0.77 ± 0.01 |
| ARC-f-Max-RKL (Our) | **-1.43 ± 0.08** | -2.91 ± 0.25 | **-0.38 ± 0.02** | **-1.25 ± 0.06** |
| f-Max-RKL | -2.22 ± 0.09 | -3.38 ± 0.15 | -0.8 ± 0.05 | -2.03 ± 0.06 |
| ARC-GAIL (Our) | -1.53 ± 0.06 | **-2.64 ± 0.07** | -0.46 ± 0.01 | -1.56 ± 0.08 |
| GAIL | -2.78 ± 0.09 | -4.53 ± 0.01 | -1.05 ± 0.06 | -2.35 ± 0.06 |

---

## 6. Limitations

Three main limitations in our work are: (1) While many AIL algorithms can be trained using expert ‘states’ only, ARC-AIL can only be trained with ‘state-action’ (s, a) pairs. There are several scenarios where obtaining (s, a) pairs is challenging (e.g. kinesthetic teaching). In such scenarios, ARC is not directly applicable. People often use tricks to mitigate this issue and using (s, a) pairs to train a policy is a popular choice [38, 39, 40, 41, 42]. (2) ARC-AIL can only work with continuous action space. Most real world robotic tasks have or can be modified to have a continuous action space. (3) We haven’t explored how the agent-adversary interaction in AIL affects the accuracy of the reward gradient and leave that for future work.

---

## 7. Conclusion

We highlighted that the reward in popular Adversarial Imitation Learning (AIL) algorithms are differentiable but this property has not been leveraged by existing model-free RL algorithms to train a policy. Further, they are usually shaped. We also showed that naively differentiating the policy through this reward function does not perform well. To solve this issue, we proposed a class of Actor Residual Critic (ARC) RL algorithms that use a C function as an alternative to standard Actor Critic (AC) algorithms which use a Q function. An ARC algorithm can replace the AC algorithm in any existing AIL algorithm. We formally proved that Policy Iteration using C function converges to an optimum policy in tabular environments. For continuous-control tasks, using ARC can compute the exact gradient of the policy through the reward function which helps improve the performance of the AIL algorithms in simulated continuous-control and simulated & real robotic manipulation tasks. Future work can explore the applicability of ARC algorithm to other scenarios which have a differentiable reward function.

---

## References

[1] V. Mnih et al. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013.
[2] J. Schulman et al. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
[3] T. Haarnoja et al. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. ICML, 2018.
[4] C. G. Atkeson and S. Schaal. Robot learning from demonstration. ICML, 1997.
[5] S. Schaal. Learning from demonstration. NIPS, 1997.
[6] B. D. Argall et al. A survey of robot learning from demonstration. Robotics and autonomous systems, 2009.
[7] P. Abbeel et al. Autonomous helicopter aerobatics through apprenticeship learning. IJRR, 2010.
[8] M. Bain and C. Sammut. A framework for behavioural cloning. Machine Intelligence, 1995.
[9] A. Y. Ng et al. Algorithms for inverse reinforcement learning. ICML, 2000.
[10] J. Ho and S. Ermon. Generative adversarial imitation learning. NIPS, 2016.
[11] J. Fu et al. Learning robust rewards with adversarial inverse reinforcement learning. arXiv:1710.11248, 2017.
[12] S. K. S. Ghasemipour et al. A divergence minimization perspective on imitation learning methods. CoRL, 2020.
[13] J. Schulman et al. Trust region policy optimization. ICML, 2015.
[14] J. Schulman et al. High-dimensional continuous control using generalized advantage estimation. arXiv:1506.02438, 2015.
[15] R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 1992.
[16] R. S. Sutton and A. G. Barto. Reinforcement learning: An introduction. MIT press, 2018.
[17] T. P. Lillicrap et al. Continuous control with deep reinforcement learning. arXiv:1509.02971, 2015.
[18] S. Fujimoto et al. Addressing function approximation error in actor-critic methods. ICML, 2018.
[19] S. Ross et al. A reduction of imitation learning and structured prediction to no-regret online learning. AISTATS, 2011.
[20] M. Laskey et al. Iterative noise injection for scalable imitation learning. CoRL, 2017.
[21] B. D. Ziebart et al. Maximum entropy inverse reinforcement learning. AAAI, 2008.
[22] I. J. Goodfellow et al. Generative adversarial nets.
[23] Y. Li et al. Infogail: Interpretable imitation learning from visual demonstrations. NIPS, 2017.
[24] R. Jena et al. Augmenting gail with bc for sample efficient imitation learning. CoRL, 2021.
[25] L. Ke et al. Imitation learning as f-divergence minimization. arXiv:1905.12888, 2019.
[26] J. Lin. Divergence measures based on the shannon entropy. IEEE Transactions on Information theory, 1991.
[27] X. Zhang et al. f-gail: Learning f-divergence for generative adversarial imitation learning. arXiv:2010.01207, 2020.
[28] T. Ni et al. f-irl: Inverse reinforcement learning via state marginal matching. CoRL, 2021.
[29] D. Hafner et al. Dream to control: Learning behaviors by latent imagination. ICLR, 2019.
[30] A. Bemporad et al. The explicit linear quadratic regulator for constrained systems. Automatica, 2002.
[31] D. Silver et al. Deterministic policy gradient algorithms. ICML, 2014.
[32] A. Paszke et al. Pytorch: An imperative style, high-performance deep learning library. NeurIPS, 2019.
[33] M. Abadi et al. Tensorflow: A system for large-scale machine learning. OSDI, 2016.
[34] J. W. Roberts and R. Tedrake. Signal-to-noise ratio analysis of policy gradient algorithms.
[35] G. Brockman et al. Openai gym. arXiv:1606.01540, 2016.
[36] M. Wise et al. Fetch and freight: Standard platforms for service robot applications.
[37] A. Campeau-Lecours et al. Kinova modular robot arms for service robotics applications.
[38] S. Young et al. Visual imitation made easy. CoRL, 2020.
[39] X. B. Peng et al. Learning agile robotic locomotion skills by imitating animals. arXiv:2004.00784, 2020.
[40] Y. Lu et al. Aw-opt: Learning robotic skills with imitation and reinforcement at scale. CoRL, 2021.
[41] O. Scheel et al. Urban driver: Learning to drive from real-world demonstrations using policy gradients. CoRL, 2021.
[42] R. Hoque et al. Thriftydagger: Budget-aware novelty and risk gating for interactive imitation learning. CoRL, 2022.

---

## Appendix

### A. Accuracy of gradient

#### A.1 Error in gradient of an approximate function

**Theorem 2.** The error in gradient of an approximation of a differentiable function can be arbitrarily large even if the function approximation is accurate (but not exact). Formally, for any differentiable function $f(x) : A \to B$, any small value of $\epsilon > 0$ and any large value of $D > 0$, we can have an approximation $\hat{f}(x)$ s.t. the following conditions are satisfied:

$$ |\hat{f}(x) - f(x)| \le \epsilon \quad \forall x \in A \quad \text{(Accurate approximation)} \quad (14) $$

$$ |\nabla_x \hat{f}(x) - \nabla_x f(x)| \ge D \quad \text{for some } x \in A \quad \text{(Arbitrarily large error in gradient)} \quad (15) $$

**Proof.** For any differentiable $f(x)$, $\epsilon > 0$ and $D > 0$, we can construct many examples of $\hat{f}(x)$ that satisfy the conditions in Eq. 14 and 15. Here we show just one example that satisfies the 2 conditions. Let $x_0$ be any point $x_0 \in A$. We can choose $\hat{f}(x) = f(x) + \epsilon \sin(b(x - x_0))$, where $b = \frac{2D}{\epsilon}$.
The error in function approximation is:
$$ |\hat{f}(x) - f(x)| = |\epsilon \sin b(x - x_0)| = \epsilon |\sin b(x - x_0)| \le \epsilon \quad (\because \sin(x) \in [-1, 1]) $$
Thus, $\hat{f}(x)$ satisfies Eq. 14.
The error in gradient at $x_0$ is:
$$ |\nabla_x \hat{f}(x) - \nabla_x f(x)|_{x=x_0} = |\nabla_x f(x) + \epsilon b \cos (b(x - x_0)) - \nabla_x f(x)|_{x=x_0} $$
$$ = \epsilon b |\cos (b(x_0 - x_0))| = \epsilon \frac{2D}{\epsilon} |\cos (0)| = 2D > D $$
Thus, $\hat{f}(x)$ satisfies Eq.15.

#### A.2 Decomposition in ARC leads to more accurate gradient for AIL

From Theorem 2, there is no bound on the error in gradient of an approximate function. Let $\hat{Q}$ and $\hat{C}$ denote the approximated Q and C values respectively. Even in the worst case, the gradient obtained using our proposed decomposition ($Q = r + C$) would be useful because $\nabla_a r(s, a)$ is exact.

It is possible that the immediate “environment reward” is misleading which might hurt ARC. However, the “adversary reward” is a measure of closeness between agent and expert actions. It naturally is never misleading as long as we have a reasonably trained adversary.

We perform a Signal to Noise (SNR) analysis.
Let us consider the case of a 1D environment.
1.  Signal strength of $\nabla_a r(s, a) = \mathbb{E}[(\nabla_a r(s, a))^2] = S_r$. Noise = 0.
2.  $\nabla_a \hat{C}(s, a) = \nabla_a C(s, a) + \epsilon_c$
3.  Signal strength of $\nabla_a \hat{C}(s, a) = \mathbb{E}[(\nabla_a C(s, a))^2] = S_c$
4.  SNR of $\nabla_a \hat{C}(s, a) = snr_c$. Noise strength $S_n = S_c / snr_c$.
7.  Final signal $= \nabla_a r(s, a) + \nabla_a \hat{C}(s, a) = (\nabla_a r(s, a) + \nabla_a C(s, a)) + \epsilon_c$
8.  Net signal strength $= \mathbb{E}[(\nabla_a r(s, a) + \nabla_a C(s, a))^2] = S_r + S_c + 2S_{r,C}$ where $S_{r,C} = \mathbb{E}[\nabla_a r \nabla_a C]$
9.  Net SNR $= \frac{S_r + S_c + 2S_{r,C}}{S_n} = snr_c (\frac{S_r}{S_c} + 1 + \frac{2S_{r,C}}{S_c})$

Let $snr_Q$ be the SNR in $\nabla_a \hat{Q}(s, a)$. The net SNR in $\nabla_a \hat{C}(s, a)$ is higher than $snr_Q$ if:
$$ snr_c \ge \frac{1}{\frac{S_r}{S_c} + 1 + \frac{2S_{r,C}}{S_c}} snr_Q \quad (30) $$

**Case 1: $S_{r,c} \ge 0$.** This means gradients are positively correlated. The denominator factor is $>1$, meaning even if $snr_C$ is a fraction of $snr_Q$, decomposition helps.
**Case 2: $-S_r/2 \le S_{r,c} < 0$.** Denominator is still $\ge 1$. Same conclusion.
**Case 3: $S_{r,c} < -S_r/2$.** Highly negative correlation. Decomposition only helps if $snr_c > snr_Q$.

In AIL, reward is shaped, so $snr_c$ is likely similar to $snr_Q$, and decomposition helps.

### B. Properties of C function

#### B.1 Unique optimality of C function
**Lemma 1.** There exists a unique optimum $C^*$ for any MDP.
**Proof.** From optimality of Q function: $Q^*(s, a) = r(s, a) + C^*(s, a)$. Since $Q^*$ is unique, $C^*$ must be unique.

#### B.2 Bellman backup for C function
**Lemma 2.** The recursive Bellman equation for $C^\pi$ is:
$$ C^\pi(s, a) = \gamma \sum_{s'} P(s'|s, a) \sum_{a'} \pi(a'|s') (r(s', a') + C^\pi(s', a')) $$
**Proof.** Derived by expanding $C^\pi(s_t, a_t) = \mathbb{E} \sum_{k \ge 1} \gamma^k r_{t+k}$ and factoring out $\gamma$.

#### B.3 Convergence of policy evaluation using C function
**Theorem 3.** The Bellman backup operation for policy evaluation using C function converges to the true C function, $C^\pi$.
**Proof.** We define the Bellman backup operator $F$. We prove $F$ is a contraction mapping w.r.t $\infty$ norm, i.e., $||FC_1 - FC_2||_\infty \le \gamma ||C_1 - C_2||_\infty$. This implies convergence to a fixed point.

#### B.4 Convergence of policy iteration using C function
**Theorem 1.** The policy iteration algorithm defined by Algorithm 1 converges to the optimal $C^*$ function and an optimal policy $\pi^*$.
**Proof.** Policy evaluation converges (Theorem 3). Policy improvement is same as with Q function since maximizing $r+C$ is same as maximizing $Q$. This implies convergence.

### C. Popular Algorithms

#### C.1 Policy Iteration using Q function
(Standard Algorithm provided for comparison).

### D. Additional Results

#### D.1 Visualization of Real Robot Policy Execution
Fig. 9 in text shows ARC-aided methods push the block closer to the goal (lower distance) compared to standard methods.

#### D.2 Accuracy of gradient of the proposed approach
A 1D driving environment toy example is used. Code snippet provided for policies and reward. Results show that decomposition ($r+C$) leads to lower error in estimating the gradient compared to directly estimating $Q$, even after many epochs.

### E. Experimental Details

**E.1 Grid World:** 4 directions, reward 1 at goal, $\gamma=0.9$. Validates Theorem 1.
**E.2 Mujoco:** Ant-v2, Walker-v2, HalfCheetah-v2, Hopper-v2. Used code from [28]. ARC algorithms used SARC implementation. Hyperparameters detailed (LR, $\alpha$, batch size).
**E.3 Manipulation:** Simplified FetchReach/FetchPush. Expert via proportional controller.
**E.4 Sim-to-real:** Kinova Jaco Gen 2 arm. Aruco markers for tracking. Scaling factors applied to observations and actions to map sim to real.

### F. Discussion on Results

**F.1 Mujoco:** ARC methods consistently rank higher. ARC-f-Max-RKL consistently ranked 1.
**F.2 Manipulation:** ARC methods rank 1 or 2. Shows parameter robustness as hyperparameters were not extensively tuned.
**F.3 Sim-to-real:** Consistent trend where ARC outperforms baselines.