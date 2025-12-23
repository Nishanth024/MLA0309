import math

# Parameters
max_inventory = 10
max_order = 5
gamma = 0.95
holding_cost = 1
shortage_cost = 5
fixed_order_cost = 2
unit_order_cost = 1
lambda_demand = 2  # Poisson demand mean

states = list(range(max_inventory+1))
actions = list(range(max_order+1))

# Poisson demand distribution (up to 15 units)
demand_probs = []
for d in range(15):
    p = math.exp(-lambda_demand) * (lambda_demand**d) / math.factorial(d)
    demand_probs.append(p)

def immediate_cost(state, action, demand):
    new_inventory = min(max_inventory, state + action)
    leftover = max(new_inventory - demand, 0)
    shortage = max(demand - new_inventory, 0)
    order_cost = (fixed_order_cost if action > 0 else 0) + unit_order_cost * action
    return holding_cost * leftover + shortage_cost * shortage + order_cost

# Value Iteration
V = [0.0 for _ in states]
policy = [0 for _ in states]

for iteration in range(200):  # run enough iterations to converge
    new_V = [0.0 for _ in states]
    for s in states:
        costs = []
        for a in actions:
            expected_cost = 0.0
            for d, p in enumerate(demand_probs):
                next_state = min(max_inventory, max(s+a-d, 0))
                c = immediate_cost(s, a, d) + gamma * V[next_state]
                expected_cost += p * c
            costs.append(expected_cost)
        new_V[s] = min(costs)
        policy[s] = costs.index(min(costs))
    V = new_V

# Print results
print("Optimal Value Function (Expected Cost per Inventory Level):")
for s in states:
    print(f"Inventory {s}: {V[s]:.2f}")

print("\nOptimal Ordering Policy (Units to Order per Inventory Level):")
for s in states:
    print(f"Inventory {s}: Order {policy[s]} units")
