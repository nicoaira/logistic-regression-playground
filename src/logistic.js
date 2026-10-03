// Small numeric helpers for logistic regression.

export const sigmoid = (z) => 1 / (1 + Math.exp(-z));

// Inverse of the sigmoid: the z at which g(z) = c.
export const logit = (c) => Math.log(c / (1 - c));

// Average logistic loss J (Unit 6.4), clipped so a perfect 0 or 1 never gives log(0).
export const cost = (probs, ys) => {
    const eps = 1e-12;
    let total = 0;
    probs.forEach((f, i) => {
        const p = Math.min(Math.max(f, eps), 1 - eps);
        total += ys[i] === 1 ? -Math.log(p) : -Math.log(1 - p);
    });
    return total / probs.length;
};

// One gradient-descent step for z = w·x + b (Unit 6.5). xs: array of feature arrays.
export const gdStep = (w, b, xs, ys, alpha) => {
    const m = ys.length;
    const dw = new Array(w.length).fill(0);
    let db = 0;
    for (let i = 0; i < m; i++) {
        let z = b;
        for (let j = 0; j < w.length; j++) z += w[j] * xs[i][j];
        const err = sigmoid(z) - ys[i];
        for (let j = 0; j < w.length; j++) dw[j] += err * xs[i][j];
        db += err;
    }
    const newW = w.map((wj, j) => wj - (alpha * dw[j]) / m);
    const grad = Math.sqrt(dw.reduce((s, d) => s + (d / m) ** 2, 0) + (db / m) ** 2);
    return { w: newW, b: b - (alpha * db) / m, grad };
};

// Points where the line a·x + c·y = t crosses the box [x0,x1]×[y0,y1] (at most 2 kept).
export const lineInBox = (a, c, t, [x0, x1], [y0, y1]) => {
    const pts = [];
    if (Math.abs(c) > 1e-12) {
        for (const x of [x0, x1]) {
            const y = (t - a * x) / c;
            if (y >= y0 - 1e-9 && y <= y1 + 1e-9) pts.push([x, y]);
        }
    }
    if (Math.abs(a) > 1e-12) {
        for (const y of [y0, y1]) {
            const x = (t - c * y) / a;
            if (x > x0 + 1e-9 && x < x1 - 1e-9) pts.push([x, y]);
        }
    }
    return pts.slice(0, 2);
};

// Part of the box where sign·(a·x + c·y − t) ≥ 0, as a closed polygon (Sutherland–Hodgman, one edge).
export const halfPlaneInBox = (a, c, t, sign, [x0, x1], [y0, y1]) => {
    const corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    const inside = ([x, y]) => sign * (a * x + c * y - t) >= 0;
    const out = [];
    for (let i = 0; i < corners.length; i++) {
        const P = corners[i];
        const Q = corners[(i + 1) % corners.length];
        const pin = inside(P);
        const qin = inside(Q);
        if (pin) out.push(P);
        if (pin !== qin) {
            const fp = a * P[0] + c * P[1] - t;
            const fq = a * Q[0] + c * Q[1] - t;
            const s = fp / (fp - fq);
            out.push([P[0] + s * (Q[0] - P[0]), P[1] + s * (Q[1] - P[1])]);
        }
    }
    if (out.length) out.push(out[0]);
    return out;
};

export const range = (lo, hi, n) => Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1));
