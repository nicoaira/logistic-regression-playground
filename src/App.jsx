import React, { useEffect, useState } from 'react';
import { Sun, Moon } from 'lucide-react';
import OneFeature from './OneFeature';
import TwoFeatures from './TwoFeatures';

export default function App() {
    // The URL hash selects the tab, so a link can open Step 2 directly (…/#two-features).
    const [step, setStepState] = useState(() => (window.location.hash === '#two-features' ? 2 : 1));
    const setStep = (s) => {
        setStepState(s);
        window.history.replaceState(null, '', s === 2 ? '#two-features' : '#one-feature');
    };
    const [theme, setTheme] = useState('dark');

    useEffect(() => {
        document.documentElement.setAttribute('data-theme', theme);
    }, [theme]);

    const dark = theme === 'dark';
    const pTheme = {
        fontColor: dark ? '#e2e8f0' : '#1e293b',
        gridColor: dark ? '#334155' : '#cbd5e1',
        midColor: dark ? '#cbd5e1' : '#f1f5f9',
        paneColor: dark ? 'rgba(15, 23, 42, 0.6)' : 'rgba(241, 245, 249, 0.9)',
        labelBg: dark ? 'rgba(15, 23, 42, 0.85)' : 'rgba(255, 255, 255, 0.9)',
    };

    return (
        <div className="app-container">
            <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem' }}>
                <div>
                    <div className="title">Logistic Regression Playground</div>
                    <div className="subtitle">Probabilities, thresholds and decision boundaries</div>
                </div>
                <button
                    className="btn btn-secondary"
                    onClick={() => setTheme(dark ? 'light' : 'dark')}
                    style={{ padding: '0.6rem', borderRadius: '50%', width: '42px', height: '42px', justifyContent: 'center', flexShrink: 0 }}
                    title={`Switch to ${dark ? 'light' : 'dark'} mode`}
                >
                    {dark ? <Sun size={20} /> : <Moon size={20} />}
                </button>
            </header>

            <div className="nav-tabs">
                <div className={`nav-tab ${step === 1 ? 'active' : ''}`} onClick={() => setStep(1)}>
                    Step 1: One Feature (Hours Studied)
                </div>
                <div className={`nav-tab ${step === 2 ? 'active' : ''}`} onClick={() => setStep(2)}>
                    Step 2: Two Features (Studied &amp; Slept)
                </div>
            </div>

            {step === 1 ? <OneFeature pTheme={pTheme} /> : <TwoFeatures pTheme={pTheme} />}

            <div className="footer">
                Data: 40 invented exam students (hours studied, hours slept, pass or fail). The model is f = g(z) with
                g(z) = 1 / (1 + e<sup>−z</sup>); a student is predicted to pass when f ≥ c.
            </div>
        </div>
    );
}
