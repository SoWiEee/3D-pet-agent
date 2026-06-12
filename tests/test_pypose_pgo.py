"""§14.6.2 — PyPose pose-graph optimisation primitive (the SLAM back-end core)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pp = pytest.importorskip("pypose")


def test_lm_closes_a_loop() -> None:
    class PoseGraph(torch.nn.Module):
        def __init__(self, nodes):
            super().__init__()
            self.nodes = pp.Parameter(nodes)

        def forward(self, edges, meas):
            ti = self.nodes[edges[:, 0]]
            tj = self.nodes[edges[:, 1]]
            pred = ti.Inv() @ tj
            return (meas.Inv() @ pred).Log().tensor().view(-1)

    graph = PoseGraph(pp.identity_SE3(4))
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])

    def x(dx):
        return pp.SE3(torch.tensor([dx, 0, 0, 0, 0, 0, 1.0])).tensor()

    meas = pp.SE3(torch.stack([x(1.0), x(1.0), x(1.0), x(-3.0)]))
    opt = pp.optim.LM(graph)
    loss = None
    for _ in range(15):
        loss = opt.step((edges, meas))
    assert float(loss) < 1e-6
    xs = graph.nodes.tensor()[:, 0].tolist()
    assert abs((xs[1] - xs[0]) - 1.0) < 1e-2
    assert abs((xs[3] - xs[0]) - 3.0) < 1e-2
