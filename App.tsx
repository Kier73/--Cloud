import React, { useState, useEffect, useCallback, useRef } from 'react';
import { SystemState, SimulationLog, OperationMetrics, FileData, NodeConfig } from './types';
import TerminalLog from './components/TerminalLog';
import MatrixCanvas from './components/MatrixCanvas';
import PerformanceChart from './components/PerformanceChart';
import ApiDashboard from './components/ApiDashboard';
import { analyzeOperation } from './services/geminiService';
import { Activity, Cpu, HardDrive, Zap, Play, Terminal, FileArchive, UploadCloud, Download, RefreshCw, Settings, Network } from 'lucide-react';

const uid = () => Math.random().toString(36).substring(2, 9);
const now = () => new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

export default function App() {
  const [state, setState] = useState<SystemState>(SystemState.IDLE);
  const [mode, setMode] = useState<'BENCHMARK' | 'FILE' | 'API'>('BENCHMARK');
  
  // Benchmark Config
  const [matrixSize, setMatrixSize] = useState<number>(4096);
  
  // File Config
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [compressedBlob, setCompressedBlob] = useState<Blob | null>(null);
  
  // Network/API Config
  const [nodeConfig, setNodeConfig] = useState<NodeConfig>({
    nodeId: crypto.randomUUID(),
    port: 8080,
    endpoint: '/api/v1/compute',
    active: false
  });

  // System Data
  const [logs, setLogs] = useState<SimulationLog[]>([]);
  const [metrics, setMetrics] = useState<OperationMetrics | null>(null);
  const [history, setHistory] = useState<OperationMetrics[]>([]);
  const [aiAnalysis, setAiAnalysis] = useState<string>("");

  const addLog = useCallback((message: string, type: SimulationLog['type'] = 'info') => {
    setLogs(prev => [...prev, { id: uid(), timestamp: now(), message, type }]);
  }, []);

  useEffect(() => {
    addLog("Φ-Cloud Engine Online.", "system");
    addLog("Waiting for input command...", "info");
  }, [addLog]);

  // Network Listener Logic
  useEffect(() => {
    if (mode === 'API' && nodeConfig.active) {
        setState(SystemState.LISTENING);
        addLog(`Node Active. Listening on port ${nodeConfig.port}...`, "network");
    } else if (mode === 'API' && !nodeConfig.active) {
        setState(SystemState.IDLE);
        addLog("Node offline.", "system");
    }
  }, [mode, nodeConfig.active, addLog]);


  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (mode === 'API') return; // Disable drop in API mode
    if (state !== SystemState.IDLE && state !== SystemState.COMPLETED) return;

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setMode('FILE');
      setCompressedBlob(null);
      setMetrics(null);
      setAiAnalysis("");
      addLog(`File loaded: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`, "system");
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
        const file = e.target.files[0];
        setSelectedFile(file);
        setMode('FILE');
        setCompressedBlob(null);
        setMetrics(null);
        setAiAnalysis("");
        addLog(`File selected: ${file.name}`, "system");
    }
  };

  const runBenchmark = async () => {
    setState(SystemState.INITIALIZING);
    setAiAnalysis("");
    addLog(`Initializing Benchmark (N=${matrixSize})...`, "system");

    // Calculation: Elements * 4 bytes
    const elements = matrixSize * matrixSize;
    const sizeBytes = elements * 4;
    const memGB = sizeBytes / (1024 ** 3);

    addLog(`Allocating virtual buffer: ${memGB.toFixed(4)} GB`, "info");
    
    await new Promise(r => setTimeout(r, 600));

    setState(SystemState.PROCESSING);
    addLog("Running spectral analysis...", "info");

    const t0 = performance.now();
    // Simulate work relative to size
    const delay = Math.log2(matrixSize) * 150 + (Math.random() * 200);
    await new Promise(r => setTimeout(r, delay));
    const t1 = performance.now();

    const duration = (t1 - t0) / 1000;
    const throughput = (sizeBytes / 1024 / 1024) / duration; // MB/s

    finishOperation({
        mode: 'BENCHMARK',
        sizeLabel: `${matrixSize}x${matrixSize}`,
        rawSize: sizeBytes,
        processedSize: sizeBytes, 
        throughputMBps: throughput,
        duration
    });
  };

  const runFileCompression = async () => {
    if (!selectedFile) return;

    setState(SystemState.INITIALIZING);
    setAiAnalysis("");
    addLog(`Reading file: ${selectedFile.name}...`, "system");

    try {
        setState(SystemState.PROCESSING);
        addLog("Compressing data stream (GZIP)...", "info");

        const t0 = performance.now();
        const stream = selectedFile.stream();
        const compressedStream = stream.pipeThrough(new CompressionStream('gzip'));
        const response = new Response(compressedStream);
        const blob = await response.blob();
        const t1 = performance.now();
        
        const duration = (t1 - t0) / 1000;
        setCompressedBlob(blob);
        const throughput = (selectedFile.size / 1024 / 1024) / duration;
        const ratio = ((selectedFile.size - blob.size) / selectedFile.size) * 100;

        addLog(`Compression successful. Saved ${ratio.toFixed(2)}%`, "success");

        finishOperation({
            mode: 'FILE',
            sizeLabel: selectedFile.name,
            rawSize: selectedFile.size,
            processedSize: blob.size,
            throughputMBps: throughput,
            duration,
            compressionRatio: ratio
        });

    } catch (err) {
        console.error(err);
        setState(SystemState.ERROR);
        addLog("Compression failed.", "error");
        setTimeout(() => setState(SystemState.IDLE), 2000);
    }
  };

  const simulateIncomingRequest = async () => {
    if (!nodeConfig.active) return;
    
    const randomIp = `192.168.1.${Math.floor(Math.random() * 255)}`;
    addLog(`Incoming Connection Request from ${randomIp}`, "network");
    addLog(`Handshake accepted. Receiving payload...`, "network");
    
    setState(SystemState.PROCESSING);
    
    // Simulate Data Receipt
    const virtualSize = Math.floor(Math.random() * 50) + 10; // 10-60 MB
    const virtualSizeBytes = virtualSize * 1024 * 1024;
    
    await new Promise(r => setTimeout(r, 800));
    
    addLog(`Payload Received: ${virtualSize} MB. Processing...`, "info");
    
    const t0 = performance.now();
    await new Promise(r => setTimeout(r, 1200)); // Simulate compute
    const t1 = performance.now();
    
    const duration = (t1 - t0) / 1000;
    const throughput = virtualSize / duration;

    addLog(`Task Complete. Sending response to ${randomIp}`, "success");
    
    const metric: OperationMetrics = {
        mode: 'API',
        sizeLabel: 'Remote Task',
        rawSize: virtualSizeBytes,
        processedSize: virtualSizeBytes * 0.4, // Simulate compression
        throughputMBps: throughput,
        duration,
        originIp: randomIp
    };

    setMetrics(metric);
    setHistory(prev => [...prev, metric]);
    setAiAnalysis("REMOTE TASK VERIFIED. INTEGRITY CHECK PASSED.");
    
    setState(SystemState.LISTENING); // Return to listening state
  };

  const finishOperation = async (result: OperationMetrics) => {
    setState(SystemState.FINALIZING);
    setMetrics(result);
    setHistory(prev => [...prev, result]);
    
    setState(SystemState.ANALYSIS);
    addLog("Sending telemetry to Core...", "system");
    
    const analysis = await analyzeOperation(result, logs.map(l => l.message));
    setAiAnalysis(analysis);
    
    setState(SystemState.COMPLETED);
    addLog("Operation confirmed. Ready.", "system");
  };

  const handleDownload = () => {
    if (!compressedBlob || !selectedFile) return;
    const url = URL.createObjectURL(compressedBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedFile.name}.gz`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    addLog("File downloaded.", "info");
  };

  const reset = () => {
    setState(SystemState.IDLE);
    setMetrics(null);
    setCompressedBlob(null);
    setAiAnalysis("");
  };

  const isBusy = (state === SystemState.PROCESSING || state === SystemState.INITIALIZING || state === SystemState.FINALIZING || state === SystemState.ANALYSIS);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-4 md:p-8 font-sans selection:bg-emerald-500/30">
      
      {/* HEADER */}
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tighter text-white flex items-center gap-2">
            <Cpu className="text-emerald-500" /> Φ-Cloud <span className="text-slate-600 font-light">| Compute Engine</span>
          </h1>
          <p className="text-slate-500 text-sm mt-1 font-mono">Holographic Data Interface // Node: LOCAL_HOST</p>
        </div>
        <div className="mt-4 md:mt-0 flex gap-4">
            <div className="flex flex-col items-end">
                <span className="text-xs text-slate-500 uppercase tracking-widest">Status</span>
                <span className={`font-mono font-bold ${
                    state === SystemState.IDLE || state === SystemState.COMPLETED ? 'text-slate-400' : 
                    state === SystemState.ERROR ? 'text-rose-500' :
                    state === SystemState.LISTENING ? 'text-blue-400 animate-pulse' :
                    'text-emerald-400 animate-pulse'
                }`}>
                    {state}
                </span>
            </div>
        </div>
      </header>

      <main className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-full">
        
        {/* LEFT COLUMN: Controls & Logs */}
        <div className="lg:col-span-4 flex flex-col gap-6">
            
            {/* Control Deck */}
            <div className="bg-slate-900 border border-slate-700 rounded-lg p-6 shadow-lg">
                <div className="flex items-center gap-2 mb-6 text-emerald-500 border-b border-slate-800 pb-2">
                    <Activity size={18} />
                    <h2 className="text-sm font-bold uppercase tracking-widest">Control Deck</h2>
                </div>

                {/* Mode Switcher */}
                <div className="flex bg-slate-950 rounded p-1 mb-6 border border-slate-800">
                    <button 
                        onClick={() => { setMode('BENCHMARK'); reset(); setNodeConfig(c => ({...c, active: false})); }}
                        disabled={isBusy}
                        className={`flex-1 py-2 text-[10px] md:text-xs font-bold uppercase rounded transition-colors ${mode === 'BENCHMARK' ? 'bg-slate-800 text-white' : 'text-slate-500 hover:text-slate-300'}`}
                    >
                        Benchmark
                    </button>
                    <button 
                        onClick={() => { setMode('FILE'); reset(); setNodeConfig(c => ({...c, active: false})); }}
                        disabled={isBusy}
                        className={`flex-1 py-2 text-[10px] md:text-xs font-bold uppercase rounded transition-colors ${mode === 'FILE' ? 'bg-slate-800 text-white' : 'text-slate-500 hover:text-slate-300'}`}
                    >
                        File
                    </button>
                    <button 
                        onClick={() => { setMode('API'); reset(); }}
                        disabled={isBusy}
                        className={`flex-1 py-2 text-[10px] md:text-xs font-bold uppercase rounded transition-colors flex items-center justify-center gap-1 ${mode === 'API' ? 'bg-blue-900 text-blue-100 border border-blue-700' : 'text-slate-500 hover:text-slate-300'}`}
                    >
                        <Network size={12} /> Network
                    </button>
                </div>
                
                <div className="space-y-6">
                    
                    {/* BENCHMARK CONTROLS */}
                    {mode === 'BENCHMARK' && (
                        <div className="animate-in fade-in duration-300">
                            <label className="flex justify-between text-slate-400 text-xs uppercase mb-2">
                                <span>Matrix Buffer Size</span>
                                <span className="text-emerald-400 font-mono">{matrixSize} x {matrixSize}</span>
                            </label>
                            <input 
                                type="range" 
                                min="1024" 
                                max="16384" 
                                step="1024"
                                value={matrixSize}
                                onChange={(e) => setMatrixSize(parseInt(e.target.value))}
                                disabled={isBusy}
                                className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                            />
                            <div className="flex justify-between text-[10px] text-slate-600 font-mono mt-1">
                                <span>1K</span>
                                <span>4K</span>
                                <span>8K</span>
                                <span>16K</span>
                            </div>
                        </div>
                    )}

                    {/* FILE CONTROLS */}
                    {mode === 'FILE' && (
                        <div 
                            className={`animate-in fade-in duration-300 border-2 border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer
                                ${selectedFile ? 'border-emerald-500/50 bg-emerald-900/10' : 'border-slate-700 bg-slate-950 hover:border-slate-500'}
                            `}
                            onDragOver={handleDragOver}
                            onDrop={handleDrop}
                            onClick={() => document.getElementById('fileInput')?.click()}
                        >
                            <input 
                                id="fileInput" 
                                type="file" 
                                className="hidden" 
                                onChange={handleFileSelect} 
                            />
                            
                            {selectedFile ? (
                                <div className="flex flex-col items-center">
                                    <FileArchive className="text-emerald-400 mb-2" size={32} />
                                    <div className="text-sm text-white font-mono break-all">{selectedFile.name}</div>
                                    <div className="text-xs text-slate-500 mt-1">{(selectedFile.size / 1024).toFixed(2)} KB</div>
                                    <div className="text-[10px] text-emerald-500 mt-2 uppercase">Ready for Compression</div>
                                </div>
                            ) : (
                                <div className="flex flex-col items-center py-4">
                                    <UploadCloud className="text-slate-500 mb-2" size={32} />
                                    <div className="text-sm text-slate-400">Drag file here or click</div>
                                    <div className="text-xs text-slate-600 mt-1">Supports GZIP Stream</div>
                                </div>
                            )}
                        </div>
                    )}
                    
                    {/* API CONTROLS */}
                    {mode === 'API' && (
                        <div className="animate-in fade-in duration-300 p-4 bg-slate-950 rounded border border-slate-800">
                            <div className="flex justify-between items-center mb-4">
                                <span className="text-slate-400 text-xs uppercase">Server Status</span>
                                <button
                                    onClick={() => setNodeConfig(c => ({...c, active: !c.active}))}
                                    className={`w-12 h-6 rounded-full relative transition-colors ${nodeConfig.active ? 'bg-blue-600' : 'bg-slate-700'}`}
                                >
                                    <div className={`absolute top-1 left-1 w-4 h-4 rounded-full bg-white transition-transform ${nodeConfig.active ? 'translate-x-6' : 'translate-x-0'}`} />
                                </button>
                            </div>
                            <div className="text-[10px] text-slate-500 font-mono">
                                {nodeConfig.active 
                                    ? "Accepting connections on port 8080..." 
                                    : "Server offline. Toggle to start listening."}
                            </div>
                        </div>
                    )}

                    {/* ACTION BUTTON */}
                    {(mode !== 'API') && (
                        <button
                            onClick={mode === 'BENCHMARK' ? runBenchmark : runFileCompression}
                            disabled={isBusy || (mode === 'FILE' && !selectedFile)}
                            className={`w-full py-4 rounded font-bold tracking-widest transition-all duration-300 flex items-center justify-center gap-2
                                ${isBusy || (mode === 'FILE' && !selectedFile)
                                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                                    : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-[0_0_20px_rgba(16,185,129,0.3)]' 
                                }
                            `}
                        >
                            {isBusy ? (
                                <> <Zap size={18} className="animate-spin" /> PROCESSING... </>
                            ) : (
                                <> <Play size={18} /> {mode === 'BENCHMARK' ? 'RUN BENCHMARK' : 'COMPRESS FILE'} </>
                            )}
                        </button>
                    )}
                    
                    {compressedBlob && (
                         <button
                            onClick={handleDownload}
                            className="w-full py-2 bg-slate-700 hover:bg-slate-600 text-white rounded font-bold text-xs uppercase tracking-widest flex items-center justify-center gap-2 transition-colors"
                        >
                            <Download size={14} /> Download Archive
                        </button>
                    )}
                </div>
            </div>

            {/* System Logs */}
            <div className="flex-grow min-h-[250px] lg:h-auto">
                <TerminalLog logs={logs} />
            </div>
        </div>

        {/* CENTER/RIGHT: Visuals & Data */}
        <div className="lg:col-span-8 flex flex-col gap-6">
            
            {/* Top Row: Visualizer & Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="relative">
                     <MatrixCanvas 
                        state={state} 
                        label={mode === 'API' ? 'INCOMING_TRAFFIC' : mode === 'FILE' && selectedFile ? 'DATA_STREAM' : `${matrixSize}x${matrixSize}_BUFFER`} 
                    />
                </div>
                
                <div className="flex flex-col gap-6 h-[400px] md:h-auto">
                    {mode === 'API' ? (
                        <ApiDashboard 
                            config={nodeConfig} 
                            onSimulateRequest={simulateIncomingRequest} 
                            isListening={nodeConfig.active}
                        />
                    ) : (
                        <div className="bg-slate-900 border border-slate-700 rounded-lg p-6 flex-grow flex flex-col">
                             <div className="flex items-center gap-2 mb-4 text-cyan-500">
                                <HardDrive size={18} />
                                <h2 className="text-sm font-bold uppercase tracking-widest">Telemetry</h2>
                            </div>
                            
                            {metrics ? (
                                <div className="space-y-4">
                                    <div className="flex justify-between items-end border-b border-slate-800 pb-2">
                                        <div className="text-slate-500 text-xs uppercase">Operation Time</div>
                                        <div className="text-xl font-mono text-white">{metrics.duration.toFixed(3)}s</div>
                                    </div>
                                    <div className="flex justify-between items-end border-b border-slate-800 pb-2">
                                        <div className="text-slate-500 text-xs uppercase">Throughput</div>
                                        <div className="text-xl font-mono text-emerald-400">{metrics.throughputMBps.toFixed(2)} <span className="text-sm">MB/s</span></div>
                                    </div>
                                    {metrics.mode === 'FILE' && (
                                        <>
                                            <div className="flex justify-between items-end border-b border-slate-800 pb-2">
                                                <div className="text-slate-500 text-xs uppercase">Compression Ratio</div>
                                                <div className="text-xl font-mono text-purple-400">{metrics.compressionRatio?.toFixed(1)}%</div>
                                            </div>
                                            <div className="flex justify-between items-end">
                                                <div className="text-slate-500 text-xs uppercase">Final Size</div>
                                                <div className="text-xl font-mono text-white">{(metrics.processedSize / 1024).toFixed(2)} KB</div>
                                            </div>
                                        </>
                                    )}
                                </div>
                            ) : (
                                <div className="flex-grow flex items-center justify-center flex-col gap-2 text-slate-600 font-mono text-sm opacity-50">
                                    <RefreshCw size={24} />
                                    <span>SYSTEM IDLE</span>
                                </div>
                            )}
                            
                            {/* Embedded Gemini Core Output for Non-API modes */}
                             <div className="mt-auto pt-6 border-t border-slate-800">
                                 <div className="flex items-center gap-2 mb-2 text-purple-400">
                                    <Terminal size={14} />
                                    <h2 className="text-xs font-bold uppercase tracking-widest">Analysis</h2>
                                </div>
                                <div className="font-mono text-xs leading-relaxed text-purple-100/90 min-h-[40px]">
                                    {aiAnalysis || <span className="text-slate-600 italic">Waiting for data...</span>}
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Bottom Row: Charts */}
            <div className="w-full">
                <PerformanceChart history={history} />
            </div>

        </div>
      </main>
    </div>
  );
}