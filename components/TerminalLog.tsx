import React, { useEffect, useRef } from 'react';
import { SimulationLog } from '../types';

interface TerminalLogProps {
  logs: SimulationLog[];
}

const TerminalLog: React.FC<TerminalLogProps> = ({ logs }) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div className="flex flex-col h-64 md:h-full bg-slate-900 border border-slate-700 rounded-lg p-4 font-mono text-xs md:text-sm overflow-hidden shadow-inner">
      <div className="flex justify-between items-center mb-2 border-b border-slate-700 pb-2">
        <span className="text-slate-400 uppercase tracking-widest">System Output</span>
        <div className="flex space-x-2">
            <div className="w-2 h-2 rounded-full bg-red-500"></div>
            <div className="w-2 h-2 rounded-full bg-yellow-500"></div>
            <div className="w-2 h-2 rounded-full bg-green-500"></div>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto space-y-1 pr-2">
        {logs.map((log) => (
          <div key={log.id} className="flex">
            <span className="text-slate-500 mr-2 min-w-[70px]">[{log.timestamp}]</span>
            <span className={`${
              log.type === 'error' ? 'text-rose-500 font-bold' :
              log.type === 'success' ? 'text-emerald-400' :
              log.type === 'warning' ? 'text-amber-400' :
              log.type === 'system' ? 'text-cyan-400' :
              'text-slate-300'
            }`}>
              {log.type === 'system' && '> '}
              {log.message}
            </span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
};

export default TerminalLog;
