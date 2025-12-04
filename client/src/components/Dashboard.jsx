import React, { useState } from 'react';
import { Activity, Cpu, AlertTriangle, FileText } from 'lucide-react';
import { motion } from 'framer-motion';
import InputForm from './InputForm';
import ResultsDisplay from './ResultsDisplay';
import axios from 'axios';

const Dashboard = () => {
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleAnalyze = async (observation) => {
        setLoading(true);
        setError(null);
        setResults(null);
        try {
            const response = await axios.post('http://localhost:5000/api/analyze', {
                observation: observation
            });
            setResults(response.data);
        } catch (err) {
            console.error("Analysis failed:", err);
            setError(err.response?.data?.error || "Failed to connect to the server.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container">
            <header className="mb-8 text-center">
                <motion.h1
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-4xl font-bold mb-2 gradient-text"
                >
                    Predictive Maintenance System
                </motion.h1>
                <p className="text-secondary">AI-Powered Agentic Analysis & Diagnosis</p>
            </header>

            <div className="grid gap-8">
                <InputForm onAnalyze={handleAnalyze} isLoading={loading} />

                {error && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="p-4 bg-red-900/20 border border-red-500/50 rounded-lg text-red-200 text-center"
                    >
                        {error}
                    </motion.div>
                )}

                {results && <ResultsDisplay results={results} />}
            </div>
        </div>
    );
};

export default Dashboard;
