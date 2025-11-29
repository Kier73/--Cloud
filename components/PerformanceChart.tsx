import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { OperationMetrics } from '../types';

interface PerformanceChartProps {
  history: OperationMetrics[];
}

const PerformanceChart: React.FC<PerformanceChartProps> = ({ history }) => {
  if (history.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-slate-500 text-sm font-mono border border-slate-800 rounded-lg bg-slate-900/50">
        NO PERFORMANCE DATA AVAILABLE
      </div>
    );
  }

  const data = history.map((h, i) => ({
    name: i + 1,
    throughput: h.throughputMBps,
    mode: h.mode
  }));

  return (
    <div className="h-64 w-full bg-slate-900/50 border border-slate-700 rounded-lg p-4">
      <div className="text-slate-400 text-xs uppercase tracking-widest mb-4">Throughput (MB/s)</div>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis 
            dataKey="name" 
            stroke="#64748b" 
            tick={{fontSize: 10}}
          />
          <YAxis 
            stroke="#64748b" 
            tick={{fontSize: 10}}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#e2e8f0' }}
            itemStyle={{ color: '#10b981' }}
            labelStyle={{ color: '#94a3b8' }}
            formatter={(value: number) => [`${value.toFixed(2)} MB/s`, 'Throughput']}
            labelFormatter={(label) => `Run #${label}`}
          />
          <Line 
            type="monotone" 
            dataKey="throughput" 
            stroke="#10b981" 
            strokeWidth={2}
            dot={{ r: 4, fill: '#10b981' }}
            activeDot={{ r: 6, fill: '#34d399' }}
            animationDuration={500}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PerformanceChart;