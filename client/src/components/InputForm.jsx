import React, { useState } from 'react';
import { Play, RotateCcw } from 'lucide-react';
import { motion } from 'framer-motion';

const InputForm = ({ onAnalyze, isLoading }) => {
    // Default sample data (FD001 sample)
    const defaultObservation = "-0.0007, -0.0004, 100.0, 518.67, 641.82, 1589.70, 1400.60, 14.62, 21.61, 554.36, 2388.06, 9046.19, 1.30, 47.47, 521.66, 2388.02, 8138.62, 8.4195, 0.03, 392, 2388, 100.00, 39.06, 23.4190";

    const [input, setInput] = useState(defaultObservation);

    const handleSubmit = (e) => {
        e.preventDefault();
        // Convert string to array of floats
        try {
            const observation = input.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
            if (observation.length === 0) {
                alert("Please enter valid sensor data.");
                return;
            }
            onAnalyze(observation);
        } catch (err) {
            alert("Invalid format. Please use comma-separated numbers.");
        }
    };

    const handleReset = () => {
        setInput(defaultObservation);
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-panel p-6"
        >
            <div className="flex justify-between items-center mb-4">
                <div>
                    <h2 className="text-xl font-semibold text-white">Sensor Observation</h2>
                    <p className="text-xs text-slate-400 mt-1">Enter 24 comma-separated float values</p>
                </div>
                <button
                    onClick={handleReset}
                    className="text-sm text-slate-400 hover:text-white flex items-center gap-1 transition-colors"
                >
                    <RotateCcw size={14} /> Reset Default
                </button>
            </div>

            <form onSubmit={handleSubmit}>
                <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    className="w-full h-32 bg-slate-800/50 border border-slate-600 rounded-lg p-3 text-slate-200 font-mono text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none resize-none transition-all"
                    placeholder="Enter comma-separated sensor readings..."
                />

                <div className="mt-4 flex justify-end">
                    <button
                        type="submit"
                        disabled={isLoading}
                        className={`
              flex items-center gap-2 px-6 py-2 rounded-lg font-medium text-white transition-all
              ${isLoading
                                ? 'bg-slate-600 cursor-not-allowed'
                                : 'bg-blue-600 hover:bg-blue-500 shadow-lg shadow-blue-500/20 active:scale-95'}
            `}
                    >
                        {isLoading ? (
                            <>
                                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                Analyzing...
                            </>
                        ) : (
                            <>
                                <Play size={18} />
                                Run Analysis
                            </>
                        )}
                    </button>
                </div>
            </form>
        </motion.div>
    );
};

export default InputForm;
