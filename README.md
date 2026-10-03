# Logistic Regression Playground

Interactive companion to Module 6 (classification with logistic regression) of an introductory machine-learning course.

**Live page:** https://nicoaira.github.io/logistic-regression-playground/

- **Step 1 — one feature (hours studied).** The sigmoid curve f(x) = g(w·x + b) with a movable threshold c, and the
  input line where the decision boundary is a single point x* = (logit(c) − b) / w.
- **Step 2 — two features (hours studied, hours slept).** The probability surface in 3D with a threshold plane at
  height c, and the input plane from above. Moving c slides the boundary w₁x₁ + w₂x₂ + b = logit(c); only w₁ and w₂
  can turn it.

Both steps let you change the parameters with sliders, train the model with gradient descent from zero, or jump to
the best fit. Mistakes, accuracy and the cost J (log loss) update in real time.

Data: 40 invented students (the course's `datasets.exam()`, seed 3).

## Run locally

```bash
npm install
npm run dev
```

Pushing to `main` builds the site and deploys it to GitHub Pages (`.github/workflows/deploy.yml`).
