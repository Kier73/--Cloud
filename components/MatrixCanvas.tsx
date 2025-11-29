import React, { useRef, useEffect } from 'react';
import { SystemState } from '../types';

interface MatrixCanvasProps {
  state: SystemState;
  label: string;
}

const MatrixCanvas: React.FC<MatrixCanvasProps> = ({ state, label }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    
    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;

    const drawNoise = () => {
      const time = Date.now();
      // Increase intensity during active processing states
      const isProcessing = state === SystemState.PROCESSING || state === SystemState.INITIALIZING;
      const intensity = isProcessing ? 255 : 80;
      
      for (let i = 0; i < data.length; i += 4) {
        const val = Math.random() * intensity;
        
        if (state === SystemState.ERROR) {
            // Red for error
            data[i] = val; 
            data[i + 1] = 0;
            data[i + 2] = 0;
        } else if (state === SystemState.COMPLETED) {
            // Blue/Cyan for success/idle
            data[i] = val * 0.1;
            data[i + 1] = val * 0.8;
            data[i + 2] = val;
        } else if (state === SystemState.LISTENING) {
            // Amber/Orange pulsing for listening
            const pulse = (Math.sin(time * 0.002) + 1) * 0.5; // 0 to 1
            data[i] = val * 0.8;
            data[i + 1] = val * 0.5 * pulse;
            data[i + 2] = 0;
        } else {
            // Emerald green for standard matrix look
            data[i] = val * 0.1;
            data[i + 1] = val;
            data[i + 2] = val * 0.4;
        }
        data[i + 3] = 255;
      }
      
      ctx.putImageData(imageData, 0, 0);
      
      // Scanline
      ctx.fillStyle = "rgba(0, 0, 0, 0.1)";
      const scanLineY = (time / 2) % height;
      ctx.fillRect(0, scanLineY, width, 2);

      animationRef.current = requestAnimationFrame(drawNoise);
    };

    drawNoise();

    return () => {
      cancelAnimationFrame(animationRef.current);
    };
  }, [state]);

  return (
    <div className="relative w-full aspect-square bg-black border border-slate-700 rounded-lg overflow-hidden shadow-[0_0_15px_rgba(16,185,129,0.1)] transition-colors duration-500">
        <canvas 
            ref={canvasRef} 
            width={256} 
            height={256} 
            className="w-full h-full rendering-pixelated"
        />
        <div className="absolute top-2 left-2 text-xs font-mono text-emerald-500 bg-black/70 px-2 py-1 rounded backdrop-blur-sm border border-emerald-900/30">
            BUFFER: {label}
        </div>
        
        {state === SystemState.PROCESSING && (
            <div className="absolute inset-0 flex items-center justify-center bg-emerald-900/20 backdrop-blur-[1px]">
                <div className="text-emerald-400 font-bold tracking-widest animate-pulse border-2 border-emerald-500 px-4 py-2 bg-black/60">
                    PROCESSING
                </div>
            </div>
        )}
        
        {state === SystemState.ERROR && (
            <div className="absolute inset-0 flex items-center justify-center bg-red-900/20 backdrop-blur-[1px]">
                <div className="text-red-500 font-bold tracking-widest border-2 border-red-500 px-4 py-2 bg-black/60">
                    OPERATION FAILED
                </div>
            </div>
        )}

        {state === SystemState.LISTENING && (
            <div className="absolute inset-0 flex items-center justify-center bg-blue-900/10">
                <div className="absolute bottom-4 right-4 flex items-center gap-2">
                    <span className="w-2 h-2 bg-blue-500 rounded-full animate-ping"></span>
                    <span className="text-[10px] text-blue-400 font-mono uppercase">Awaiting Request</span>
                </div>
            </div>
        )}
    </div>
  );
};

export default MatrixCanvas;