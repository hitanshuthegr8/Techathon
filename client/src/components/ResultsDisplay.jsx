import React from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, Clock, Activity, FileText, Settings } from 'lucide-react';

const Card = ({ title, icon: Icon, children, className = "" }) => (
    <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className={`glass-panel p-6 ${className}`}
    >
        <div className="flex items-center gap-2 mb-4 text-slate-300 border-b border-slate-700/50 pb-2">
            <Icon size={20} className="text-blue-400" />
            <h3 className="font-semibold">{title}</h3>
        </div>
        {children}
    </motion.div>
);

const RiskIndicator = ({ level, score }) => {
    const colors = {
        HIGH: "text-red-500",
        MEDIUM: "text-yellow-500",
        LOW: "text-green-500"
    };

    const bgColors = {
        HIGH: "bg-red-500",
        MEDIUM: "bg-yellow-500",
        LOW: "bg-green-500"
    };

    return (
        <div className="flex flex-col items-center justify-center py-4">
            <div className={`text-4xl font-bold ${colors[level] || "text-slate-400"} mb-2`}>
                {level} RISK
            </div>
            <div className="w-full bg-slate-700 h-4 rounded-full overflow-hidden">
                <div
                    className={`h-full ${bgColors[level] || "bg-slate-500"} transition-all duration-1000`}
                    style={{ width: `${Math.min(score * 100, 100)}%` }}
                />
            </div>
            <p className="mt-2 text-slate-400">Risk Score: {(score * 100).toFixed(1)}%</p>
        </div>
    );
};

const ResultsDisplay = ({ results }) => {
    if (!results) return null;

    const { risk_assessment, diagnosis, maintenance_schedule, final_report, predictions } = results;

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Risk Assessment */}
            <Card title="Risk Assessment" icon={AlertTriangle} className="md:col-span-1">
                <RiskIndicator
                    level={risk_assessment?.risk_level}
                    score={risk_assessment?.risk_score}
                />
                <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
                    <div className="bg-slate-800/50 p-3 rounded">
                        <span className="text-slate-400 block">RUL Estimate</span>
                        <span className="text-xl font-mono text-white">{risk_assessment?.avg_rul?.toFixed(0)} cycles</span>
                    </div>
                    <div className="bg-slate-800/50 p-3 rounded">
                        <span className="text-slate-400 block">Confidence</span>
                        <span className="text-xl font-mono text-white">
                            {diagnosis?.confidence != null
                                ? `${(diagnosis.confidence * 100).toFixed(0)}%`
                                : "N/A"}
                        </span>
                    </div>
                </div>
            </Card>

            {/* Diagnosis */}
            <Card title="Diagnosis" icon={Activity} className="md:col-span-1">
                <div className="space-y-4">
                    <div>
                        <span className="text-slate-400 text-sm">Probable Component</span>
                        <div className="text-lg font-medium text-white flex items-center gap-2">
                            <Settings size={16} />
                            {diagnosis?.probable_component}
                        </div>
                    </div>

                    <div>
                        <span className="text-slate-400 text-sm">Anomalies Detected</span>
                        <div className="flex flex-wrap gap-2 mt-1">
                            {diagnosis?.anomalies?.length > 0 ? (
                                diagnosis.anomalies.map((anomaly, idx) => (
                                    <span key={idx} className="px-2 py-1 bg-red-500/20 text-red-300 text-xs rounded border border-red-500/30">
                                        {anomaly}
                                    </span>
                                ))
                            ) : (
                                <span className="text-slate-500 italic">None detected</span>
                            )}
                        </div>
                    </div>
                </div>
            </Card>

            {/* Maintenance Schedule */}
            <Card title="Maintenance Schedule" icon={Clock} className="md:col-span-1">
                <div className={`
          p-4 rounded-lg border mb-4 text-center font-bold
          ${maintenance_schedule?.maintenance_window === 'IMMEDIATE' ? 'bg-red-900/20 border-red-500 text-red-400' :
                        maintenance_schedule?.maintenance_window === 'SOON' ? 'bg-yellow-900/20 border-yellow-500 text-yellow-400' :
                            'bg-green-900/20 border-green-500 text-green-400'}
        `}>
                    {maintenance_schedule?.maintenance_window} ACTION REQUIRED
                </div>
                <div className="space-y-2">
                    {maintenance_schedule?.recommended_actions?.map((action, idx) => (
                        <div key={idx} className="flex items-start gap-2 text-sm text-slate-300">
                            <CheckCircle size={14} className="mt-1 text-blue-400 shrink-0" />
                            {action}
                        </div>
                    ))}
                </div>
            </Card>

            {/* Final Report */}
            <Card title="Agentic Report" icon={FileText} className="md:col-span-2">
                <div className="prose prose-invert prose-sm max-w-none">
                    <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700 text-slate-300 leading-relaxed whitespace-pre-wrap">
                        {final_report?.narrative || "No report generated."}
                    </div>
                </div>
            </Card>
        </div>
    );
};

export default ResultsDisplay;
