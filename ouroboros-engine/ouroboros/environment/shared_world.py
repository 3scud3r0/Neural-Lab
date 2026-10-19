import torch


class SharedWorld:
    """Mundo 2D limitado, recursos esgotáveis e competição proporcional."""

    def __init__(self, population, device, config):
        self.config, self.device = config, device
        self.positions = torch.rand(population, 2, device=device) * config["size"]
        self.velocity = torch.zeros_like(self.positions)
        self.sources = torch.rand(config["resource_count"], 2, device=device) * config["size"]
        self.stock = torch.ones(config["resource_count"], device=device)
        self.obstacles = torch.rand(config.get("obstacle_count", 0), 2, device=device) * config["size"]
        self.obstacle_phase = self.obstacles.clone()
        self.time = 0.0

    def nearest(self):
        distances = torch.cdist(self.positions, self.sources)
        distance, index = distances.min(-1)
        return distance, index

    def observe(self, energy, stress, incoming):
        distance, index = self.nearest()
        direction = (self.sources[index] - self.positions) / self.config["size"]
        return torch.cat((self.positions / self.config["size"], direction,
                          distance[:, None] / self.config["size"], energy[:, None],
                          stress[:, None], incoming), dim=-1)

    @torch.no_grad()
    def step(self, actions, dt):
        before, _ = self.nearest()
        c = self.config
        previous = self.positions.clone()
        self.velocity = c["friction"] * self.velocity + (1 - c["friction"]) * actions
        self.positions.add_(self.velocity * c["speed"] * dt).clamp_(0, c["size"])
        self.time += dt
        collision = torch.zeros(len(self.positions), device=self.device, dtype=torch.bool)
        if len(self.obstacles):
            self.obstacles = (self.obstacle_phase + 0.2 * torch.sin(self.obstacle_phase + self.time)).clamp(0, c["size"])
            collision = (torch.cdist(self.positions, self.obstacles) < c.get("obstacle_radius", 0.35)).any(-1)
            self.positions[collision] = previous[collision]
            self.velocity[collision] = 0
        after, index = self.nearest()
        eligible = (after < c["harvest_radius"]).float()
        demand = torch.zeros_like(self.stock).scatter_add_(0, index, eligible)
        share = torch.minimum(self.stock / demand.clamp_min(1), torch.ones_like(self.stock))
        harvest = eligible * share[index]
        consumed = torch.zeros_like(self.stock).scatter_add_(0, index, harvest)
        self.stock.sub_(consumed).add_(c["resource_regeneration"]).clamp_(0, 1)
        # Recompensa explícita de aproximação, além da colheita.
        reward = harvest + 2 * (before - after) - 0.002 * actions.square().sum(-1) - 0.05 * collision.float()
        return harvest, reward

    @torch.no_grad()
    def respawn(self, mask):
        self.positions[mask] = torch.rand(int(mask.sum().item()), 2, device=self.device) * self.config["size"]
        self.velocity[mask] = 0
