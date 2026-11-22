"""Step 10 (A1): a learned estimator against the bound and the MAP fit.

Facade B, 3.5 GHz, N = 10 users per sample, one network per class (LOS / NLOS users).
Users are split at random into a training half and a held-out test half.
Sample: the wall moved by (Delta d ~ U[-6, 6] m, Delta phi ~ U[-2, 2] deg) from the map, material
drawn uniformly from 7 ITU classes; reports from the model at the true positions, sigma_e = 1 dB;
the estimator sees positions with N(0, 1 m^2) errors.
Network (DeepSets): per user [r - mu_map(u_hat), mu_map(u_hat), u_hat - b] -> MLP -> mean pool ->
MLP -> (Delta d, Delta phi, 7 material logits). mu_map is the twin's prediction with the map wall
and its recorded material (marble).
Test: RMSE of d against the mean robust CRB at the test truths and against the MAP fit
(step4_mle.fit, material free) on a subset, started (i) at 4 points around the map value and
(ii) at the network's estimate (its wall and material class, plus eps_r in {3, 8}): the hybrid
"network, then MAP polish". Outliers: |d error| > 3 sqrt(CRB). Material accuracy; inference time.
Output: artifacts/step10_learned.json, trained networks in artifacts/step10_net_<class>.pt
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import rsrp_model as rm
from step4_mle import fit, setup

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
CLASSES = ["concrete", "brick", "plasterboard", "wood", "glass", "marble", "metal"]
N, SIGMA_U = 10, 1.0
QUICK = len(sys.argv) > 1
N_TRAIN, N_TEST, N_MAP = (4000, 300, 6) if QUICK else (120000, 2000, 200)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def material_params(idx, freq):
    out = []
    for i in idx:
        a, b, c, d = rm.ITU[CLASSES[i]]
        f = freq / 1e9
        out.append([a * f ** b, math.log10(c * f ** d)])
    return torch.tensor(out)


def mu_batch(cfg, th, users, h0):
    """RSRP for a batch of samples, each with its own theta: th (B, 4), users (B, N, 3), h0 (B, N, M)."""
    f = lambda t, u, hr: rm.rsrp_mean_phase(cfg, t, u, torch.view_as_complex(hr), torch.zeros(u.shape[0]))
    return torch.func.vmap(f)(th, users, torch.view_as_real(h0))


def make(cfg, th0, U, H0, pool, B, gen, chunk=4000):
    xs, ys = [], []
    for s in range(0, B, chunk):
        b = min(chunk, B - s)
        idx = torch.stack([pool[torch.randperm(len(pool), generator=gen)[:N]] for _ in range(b)])
        users, h0 = U[idx], H0[idx]
        dd = (torch.rand(b, generator=gen) * 2 - 1) * 6.0
        dp = (torch.rand(b, generator=gen) * 2 - 1) * math.radians(2.0)
        mat = torch.randint(0, len(CLASSES), (b,), generator=gen)
        th = torch.cat([(th0[0] + dd)[:, None], (th0[1] + dp)[:, None], material_params(mat.tolist(), cfg.freq)], 1)
        r = mu_batch(cfg, th, users, h0) + cfg.sigma_e * torch.randn(b, N, cfg.n_wide, generator=gen)
        u_hat = users + torch.cat([SIGMA_U * torch.randn(b, N, 2, generator=gen), torch.zeros(b, N, 1)], 2)
        mu_map = mu_batch(cfg, th0.repeat(b, 1), u_hat, h0)
        rel = (u_hat - cfg.bs)[..., :2] / 100.0
        x = torch.cat([(r - mu_map) / 10.0, (mu_map + 80.0) / 20.0, rel], 2).float()
        xs.append(x)
        ys.append(dict(dd=dd, dp=dp, mat=mat, th=th, r=r, u_hat=u_hat, idx=idx))
    y = {k: torch.cat([q[k] for q in ys]) for k in ys[0]}
    return torch.cat(xs), y


class DeepSets(nn.Module):
    def __init__(self, din, h=256, n_cls=len(CLASSES)):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
        self.rho = nn.Sequential(nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2 + n_cls))

    def forward(self, x):
        return self.rho(self.phi(x).mean(1))


def train(x, y, epochs=60):
    torch.manual_seed(0)
    net = DeepSets(x.shape[-1]).float().to(DEV)              # rsrp_model sets float64 as the default dtype
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X = x.to(DEV)
    T = torch.stack([y["dd"] / 6.0, y["dp"] / math.radians(2.0)], 1).float().to(DEV)
    C = y["mat"].to(DEV)
    for ep in range(epochs):
        perm = torch.randperm(len(X), device=DEV)
        for i in range(0, len(X), 512):
            j = perm[i:i + 512]
            out = net(X[j])
            loss = ((out[:, :2] - T[j]) ** 2).mean() + 0.3 * nn.functional.cross_entropy(out[:, 2:], C[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return net.eval()


def map_job(job):
    """MAP fits of one test sample: (i) 4 starts around the map value; (ii) 3 starts at the
    network's estimate (its material class, then eps_r 3 and 8)."""
    torch.set_num_threads(1)
    k, r, u_hat, idx, th_true, th_net = job
    cfg, th0, U, H0, _ = setup("B", "3.5")
    rng = np.random.default_rng(100 + k)
    r, u_hat, h0 = torch.tensor(r), torch.tensor(u_hat), H0[torch.tensor(idx)]

    def best_of(inits):
        t = time.process_time()
        best = (float("inf"), None)
        for init in inits:
            th, c = fit(cfg, r, u_hat, h0, SIGMA_U, init, False)
            if c < best[0] and bool(torch.isfinite(th).all()):           # NaN cost or theta = failed fit
                best = (c, th)
        if best[1] is None:
            return None, time.process_time() - t
        return (best[1] - torch.tensor(th_true)).numpy(), time.process_time() - t

    around = []
    for s in range(4):
        init = th0.clone()
        init[0] += 2.0 * rng.standard_normal()
        init[1] += math.radians(1.0) * rng.standard_normal()
        init[2], init[3] = [3.0, 5.0, 7.0, 9.0][s], -1.5
        around.append(init)
    from_net = [torch.tensor(th_net)]
    for eps in [3.0, 8.0]:
        init = torch.tensor(th_net)
        init[2], init[3] = eps, -1.5
        from_net.append(init)
    a, h = best_of(around), best_of(from_net)
    # all network starts failed: the hybrid falls back to the map starts (counted in the results)
    return a, (h if h[0] is not None else (a[0], a[1] + h[1])), h[0] is None


def run(cls_name):
    cfg, th0, U, H0, los = setup("B", "3.5")
    users = np.where(los if cls_name == "LOS" else ~los)[0]
    rng = np.random.default_rng(11)
    rng.shuffle(users)
    tr, te = torch.tensor(users[: len(users) // 2]), torch.tensor(users[len(users) // 2:])
    gen = torch.Generator().manual_seed(1 if cls_name == "LOS" else 2)
    t0 = time.time()
    xtr, ytr = make(cfg, th0, U, H0, tr, N_TRAIN, gen)
    xte, yte = make(cfg, th0, U, H0, te, N_TEST, gen)
    t_gen = time.time() - t0
    net = train(xtr, ytr, epochs=8 if QUICK else 60)
    t_train = time.time() - t0 - t_gen
    with torch.no_grad():
        Xte = xte.to(DEV)
        net(Xte[:64])
        torch.cuda.synchronize() if DEV == "cuda" else None
        t1 = time.time()
        out = net(Xte).cpu()
        torch.cuda.synchronize() if DEV == "cuda" else None
        t_inf = (time.time() - t1) / len(Xte)
    err_d = out[:, 0].double() * 6.0 - yte["dd"]
    err_p = out[:, 1].double() * math.radians(2.0) - yte["dp"]
    acc = (out[:, 2:].argmax(1) == yte["mat"]).float().mean().item()
    # robust CRB at every test truth
    crb = []
    for k in range(N_TEST):
        J = rm.fisher_robust(cfg, yte["th"][k], U[yte["idx"][k]], H0[yte["idx"][k]], sigma_u=SIGMA_U).sum(0)
        crb.append(np.diag(np.linalg.pinv(J.numpy())))
    crb = np.array(crb)
    # MAP on a subset (material free, 4 starts around the map), in parallel
    from multiprocessing import get_context
    mat_hat = material_params(out[:N_MAP, 2:].argmax(1).tolist(), cfg.freq)
    th_net = torch.stack([th0[0] + out[:N_MAP, 0].double() * 6.0, th0[1] + out[:N_MAP, 1].double() * math.radians(2.0),
                          mat_hat[:, 0], mat_hat[:, 1]], 1)
    jobs = [(k, yte["r"][k].numpy(), yte["u_hat"][k].numpy(), yte["idx"][k].numpy(), yte["th"][k].numpy(),
             th_net[k].numpy()) for k in range(N_MAP)]
    t1 = time.time()
    with get_context("spawn").Pool(16) as p:          # fork breaks autograd after training
        map_err = p.map(map_job, jobs, chunksize=1)
    n_fallback = int(sum(m[2] for m in map_err))
    ok = [k for k, m in enumerate(map_err) if m[0][0] is not None]       # drop sets where every fit failed
    map_err = [map_err[k] for k in ok]
    t_map, t_hyb = np.mean([m[0][1] for m in map_err]), np.mean([m[1][1] for m in map_err])
    hyb_err = np.array([m[1][0] for m in map_err])
    map_err = np.array([m[0][0] for m in map_err])
    torch.save(net.state_dict(), os.path.join(ART, f"step10_net_{cls_name}" + ("_quick" if QUICK else "") + ".pt"))
    s3 = 3 * np.sqrt(crb[ok, 0])
    outl = lambda e: float(np.mean(np.abs(e) > s3))
    res = dict(cls=cls_name, n_train_users=len(tr), n_test_users=len(te), n_train=N_TRAIN, n_test=N_TEST,
               net_rmse_d=float(err_d.pow(2).mean().sqrt()), net_rmse_phi_deg=math.degrees(float(err_p.pow(2).mean().sqrt())),
               net_rmse_d_subset=float(err_d[:N_MAP].pow(2).mean().sqrt()),
               map_rmse_d_subset=float(np.sqrt((map_err[:, 0] ** 2).mean())),
               map_rmse_phi_deg_subset=math.degrees(float(np.sqrt((map_err[:, 1] ** 2).mean()))),
               hybrid_rmse_d_subset=float(np.sqrt((hyb_err[:, 0] ** 2).mean())),
               hybrid_rmse_phi_deg_subset=math.degrees(float(np.sqrt((hyb_err[:, 1] ** 2).mean()))),
               outliers_net=outl(err_d[ok].numpy()), outliers_map=outl(map_err[:, 0]),
               outliers_hybrid=outl(hyb_err[:, 0]), n_map=N_MAP, n_map_ok=len(map_err), n_hybrid_fallback=n_fallback,
               crb_d=float(np.sqrt(crb[:, 0].mean())), crb_phi_deg=math.degrees(float(np.sqrt(crb[:, 1].mean()))),
               crb_d_subset=float(np.sqrt(crb[:N_MAP, 0].mean())),
               net_material_acc=acc, chance=1 / len(CLASSES),
               net_ms_per_sample=1e3 * t_inf, map_s_per_sample=float(t_map), hybrid_s_per_sample=float(t_hyb),  # CPU s per sample
               gen_s=t_gen, train_s=t_train)
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    out = [run(c) for c in ["LOS", "NLOS"]]
    json.dump(out, open(os.path.join(ART, "step10_learned" + ("_quick" if QUICK else "") + ".json"), "w"), indent=1)
