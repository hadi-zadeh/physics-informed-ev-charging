# Theoretical Foundations and Analytical Bounds

This document provides formal mathematical characterizations of the continuous linear programming relaxation, projection operator properties, and analytical performance bounds established in the manuscript.

---

## Theorem 1: Exactness of the Continuous Linear Programming Relaxation

### Statement
Under standard net-billing conditions with positive retail purchasing tariffs ($\lambda_b(t) > 0$), strictly discounted solar feed-in tariffs ($\eta_s < 1$), and non-ideal battery round-trip efficiency ($\eta_{ch} \eta_{dis} < 1$), any optimal solution to the continuous relaxation strictly satisfies the binary mutual exclusivity constraints:
$$P_b^*(t) \cdot P_s^*(t) = 0 \quad 	ext{and} \quad P_{ch}^*(t) \cdot P_{dis}^*(t) = 0, \quad orall t \in \mathcal{T}$$

### Proof
1. **Grid Exchange Exclusivity:** Suppose an optimal schedule contains simultaneous import and export at interval $t$: $P_b(t) > 0$ and $P_s(t) > 0$. Let $\epsilon = \min(P_b(t), P_s(t)) > 0$. Define modified flows $P_b'(t) = P_b(t) - \epsilon$ and $P_s'(t) = P_s(t) - \epsilon$. The net power injection remains identical:
   $$P_b'(t) - P_s'(t) = (P_b(t) - \epsilon) - (P_s(t) - \epsilon) = P_b(t) - P_s(t)$$
   Hence nodal power balance is preserved. The resulting expenditure change is:
   $$\Delta C = [\lambda_b(t)(P_b - \epsilon) - \eta_s \lambda_b(t)(P_s - \epsilon)] \Delta t - [\lambda_b(t)P_b - \eta_s \lambda_b(t)P_s] \Delta t = -(1 - \eta_s) \lambda_b(t) \epsilon \Delta t < 0$$
   Since $\eta_s = 0.60 < 1$ and $\lambda_b(t) > 0$, $\Delta C < 0$, which strictly reduces cost and contradicts the optimality of simultaneous flows. Thus, $P_b^*(t) \cdot P_s^*(t) = 0$.

2. **EV Storage Exclusivity:** Simultaneous charging and discharging dissipates energy due to round-trip losses ($\eta_{ch}\eta_{dis} = 0.98 	imes 0.98 = 0.9604 < 1.0$). Net energy delivered to the battery is $\Delta E = (\eta_{ch} P_{ch} - P_{dis} / \eta_{dis}) \Delta t$. Replacing simultaneous flows with net power $P_{net} = P_{ch} - P_{dis} / (\eta_{ch}\eta_{dis})$ delivers identical $\Delta E$ while strictly reducing required imported power. Hence, $P_{ch}^*(t) \cdot P_{dis}^*(t) = 0$.

---

## Lemma 1: Firm Non-Expansiveness of 1D Interval Box Clamping

### Statement
Let $\mathcal{K} = [l, u] \subset \mathbb{R}$ be a non-empty closed interval. The projection operator $\Pi_{\mathcal{K}}(x) = \min(u, \max(l, x))$ is firmly non-expansive:
$$|\Pi_{\mathcal{K}}(x_1) - \Pi_{\mathcal{K}}(x_2)| \le |x_1 - x_2|, \quad orall x_1, x_2 \in \mathbb{R}$$

### Proof
Direct consequence of the non-expansiveness of Euclidean projections onto closed convex sets in $\mathbb{R}^1$.

---

## Proposition 1: Pre-SPP Boundary Violation Upper Bound

### Statement
Assume the training objective satisfies $\mathcal{L}_{\mathrm{total}}(	heta) \le \epsilon$. Then the expected unprojected boundary violation distance:
$$\Delta \mathrm{SoC}_{\mathrm{viol}} = \mathrm{ReLU}(\mathrm{SoC}_{t+1}^{\mathrm{raw}} - \mathrm{SoC}_{\max}) + \mathrm{ReLU}(\mathrm{SoC}_{\min} - \mathrm{SoC}_{t+1}^{\mathrm{raw}})$$
satisfies:
$$\mathbb{E}[\Delta \mathrm{SoC}_{\mathrm{viol}}] \le \sqrt{rac{2\epsilon}{\lambda_{\mathrm{boundary}}}} = \mathcal{O}\left(rac{1}{\sqrt{\lambda_{\mathrm{boundary}}}}ight)$$

### Proof
By Jensen's inequality and definition of $\mathcal{L}_{\mathrm{total}}$:
$$\lambda_{\mathrm{boundary}} (\mathbb{E}[\Delta \mathrm{SoC}_{\mathrm{viol}}])^2 \le \lambda_{\mathrm{boundary}} \mathbb{E}[(\Delta \mathrm{SoC}_{\mathrm{viol}})^2] \le \mathcal{L}_{\mathrm{total}}(	heta) \le \epsilon$$
Rearranging terms yields the stated upper bound.

---

## Proposition 2: Conditional Single-Step Regret Bound

### Statement
Suppose the single-step stage cost $c(s_t, a_t)$ is $L_c$-Lipschitz continuous with respect to action $a_t$. Under identical state observation $s_t$, the instantaneous economic regret between executed action $a(t) = \Pi_{\mathcal{K}}(	ilde{a}(t))$ and expert action $a^*(t)$ satisfies:
$$|c(s_t, a(t)) - c(s_t, a^*(t))| \le L_c \cdot (|	ilde{a}(t) - a^*(t)| + |	ilde{a}(t) - a(t)|)$$

### Proof
By the triangle inequality and $L_c$-Lipschitz continuity:
$$|c(s_t, a(t)) - c(s_t, a^*(t))| \le L_c |a(t) - a^*(t)| \le L_c (|a(t) - 	ilde{a}(t)| + |	ilde{a}(t) - a^*(t)|)$$
