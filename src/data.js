// 40 invented exam students (hours studied, hours slept the night before, result).
// Same data as the course's datasets.exam() (seed 3): a student tends to pass when
// hours studied + 0.6 × hours slept is above about 8.5, with some randomness.
const STUDIED = [1.3, 2.6, 7.7, 5.7, 1.3, 4.4, 4.8, 1.9, 7.1, 1.5, 4.0, 5.2, 4.4, 5.8, 7.1, 9.1, 3.1, 6.3, 6.8, 3.1,
    0.5, 9.3, 3.2, 3.3, 8.5, 5.8, 4.7, 7.5, 0.8, 6.9, 3.9, 1.3, 6.4, 8.9, 2.4, 6.2, 3.2, 7.2, 7.0, 2.5];
const SLEPT = [8.1, 7.1, 7.3, 8.0, 5.9, 7.7, 8.3, 4.1, 8.2, 5.7, 6.1, 4.3, 7.3, 5.1, 8.3, 5.0, 6.6, 5.7, 6.9, 4.6,
    4.5, 7.6, 7.6, 6.6, 8.6, 4.6, 8.2, 4.4, 8.8, 6.9, 6.8, 8.8, 7.8, 7.8, 3.8, 5.5, 4.0, 4.6, 4.7, 8.2];
const PASSED = [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
    0, 0, 0, 1, 1, 0];

export const STUDENTS = STUDIED.map((x1, i) => ({ x1, x2: SLEPT[i], y: PASSED[i] }));

// Maximum-likelihood fits (no regularization), checked with scikit-learn.
export const BEST_1D = { w: 1.4376, b: -7.2763 };
export const BEST_2D = { w1: 2.1701, w2: 1.2868, b: -19.3312 };

// Plot ranges. Frozen: Plotly writes zoom/pan ranges back into the arrays it is given,
// so always hand it a copy (see `copy` in ui.jsx) and keep these for the calculations.
export const X1_RANGE = Object.freeze([0, 10]);
export const X2_RANGE = Object.freeze([3, 10]);
