import React, { useState } from 'react';
import { NodeConfig } from '../types';
import { Terminal, Copy, Check, Server, Shield, Globe } from 'lucide-react';

interface ApiDashboardProps {
  config: NodeConfig;
  onSimulateRequest: () => void;
  isListening: boolean;
}

const ApiDashboard: React.FC<ApiDashboardProps> = ({ config, onSimulateRequest, isListening }) => {
  const [activeTab, setActiveTab] = useState<'PYTHON' | 'CURL' | 'NODE'>('PYTHON');
  const [copied, setCopied] = useState(false);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getCodeSnippet = () => {
    switch (activeTab) {
      case 'PYTHON':
        return `import requests
import numpy as np

# Connect to Φ-Cloud Node
NODE_URL = "http://localhost:${config.port}${config.endpoint}"
API_KEY = "phi-node-${config.nodeId.substring(0,8)}"

# Payload: 4K Matrix Float32
data = np.random.randn(4096, 4096).astype(np.float32)

print(f"Offloading {data.nbytes / 1024**2:.2f} MB to GPU Node...")

response = requests.post(
    NODE_URL, 
    headers={"X-API-Key": API_KEY},
    json={"matrix": data.tolist(), "operation": "spectral_decomposition"}
)

if response.status_code == 200:
    print("Success:", response.json())
else:
    print("Node Busy/Error")`;
      
      case 'CURL':
        return `curl -X POST http://localhost:${config.port}${config.endpoint} \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: phi-node-${config.nodeId.substring(0,8)}" \\
  -d '{
    "operation": "compression",
    "priority": "high",
    "payload_ref": "s3://bucket/dataset-large.csv"
  }'`;

      case 'NODE':
        return `const axios = require('axios');

const NODE_CONFIG = {
  url: 'http://localhost:${config.port}${config.endpoint}',
  headers: { 'X-API-Key': 'phi-node-${config.nodeId.substring(0,8)}' }
};

async function offloadCompute() {
  console.log("Connecting to Phi-Cloud Node...");
  try {
    const { data } = await axios.post(NODE_CONFIG.url, {
      task: 'holographic_reduction',
      buffer_size: 4096 * 4096
    });
    console.log("Compute Result:", data);
  } catch (err) {
    console.error("Node Unreachable");
  }
}

offloadCompute();`;
    }
    return '';
  };

  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg p-6 shadow-lg h-full flex flex-col animate-in fade-in zoom-in duration-300">
      
      {/* Header */}
      <div className="flex items-center gap-2 mb-6 text-blue-400 border-b border-slate-800 pb-2">
        <Server size={18} />
        <h2 className="text-sm font-bold uppercase tracking-widest">API Gateway Config</h2>
        <div className={`ml-auto px-2 py-0.5 rounded text-[10px] font-bold ${isListening ? 'bg-emerald-900 text-emerald-400 border border-emerald-700' : 'bg-slate-800 text-slate-500'}`}>
            {isListening ? 'PORT OPEN' : 'PORT CLOSED'}
        </div>
      </div>

      {/* Info Grid */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-slate-950 p-3 rounded border border-slate-800">
            <div className="text-xs text-slate-500 uppercase flex items-center gap-1"><Shield size={10}/> Node ID</div>
            <div className="text-sm font-mono text-white truncate" title={config.nodeId}>{config.nodeId}</div>
        </div>
        <div className="bg-slate-950 p-3 rounded border border-slate-800">
            <div className="text-xs text-slate-500 uppercase flex items-center gap-1"><Globe size={10}/> Endpoint</div>
            <div className="text-sm font-mono text-blue-400">{config.endpoint}</div>
        </div>
      </div>

      {/* Code Viewer */}
      <div className="flex-grow flex flex-col min-h-0 bg-slate-950 rounded-lg border border-slate-800 overflow-hidden">
        <div className="flex border-b border-slate-800">
            {['PYTHON', 'NODE', 'CURL'].map((lang) => (
                <button
                    key={lang}
                    onClick={() => setActiveTab(lang as any)}
                    className={`px-4 py-2 text-[10px] font-bold transition-colors ${activeTab === lang ? 'bg-slate-800 text-white border-b-2 border-blue-500' : 'text-slate-500 hover:text-slate-300'}`}
                >
                    {lang}
                </button>
            ))}
            <button 
                onClick={() => copyToClipboard(getCodeSnippet())}
                className="ml-auto px-4 text-slate-400 hover:text-white transition-colors"
                title="Copy Code"
            >
                {copied ? <Check size={14} className="text-emerald-500" /> : <Copy size={14} />}
            </button>
        </div>
        <div className="p-4 overflow-auto custom-scrollbar flex-grow relative">
            <pre className="text-xs font-mono text-slate-300 whitespace-pre-wrap">
                {getCodeSnippet()}
            </pre>
            {!isListening && (
                <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-[1px] flex items-center justify-center">
                    <div className="text-slate-500 text-xs uppercase tracking-widest border border-slate-700 px-3 py-2 rounded bg-slate-900">
                        Gateway Offline
                    </div>
                </div>
            )}
        </div>
      </div>

      {/* Simulation Trigger */}
      <div className="mt-6 pt-4 border-t border-slate-800">
        <button
            onClick={onSimulateRequest}
            disabled={!isListening}
            className={`w-full py-3 rounded font-bold tracking-widest transition-all duration-300 flex items-center justify-center gap-2 text-xs uppercase
                ${!isListening
                    ? 'bg-slate-800 text-slate-600 cursor-not-allowed'
                    : 'bg-blue-900/50 hover:bg-blue-800/50 text-blue-200 border border-blue-700/50 hover:border-blue-500' 
                }
            `}
        >
            <Terminal size={14} /> Simulate Incoming Request
        </button>
        <p className="text-[10px] text-slate-600 text-center mt-2">
            *In a production environment, this triggers a real WebSocket event.
        </p>
      </div>
    </div>
  );
};

export default ApiDashboard;