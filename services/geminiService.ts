import { GoogleGenAI } from "@google/genai";
import { OperationMetrics } from '../types';

const getClient = (): GoogleGenAI => {
  const apiKey = process.env.API_KEY || '';
  return new GoogleGenAI({ apiKey });
};

export const analyzeOperation = async (
  metrics: OperationMetrics, 
  logs: string[]
): Promise<string> => {
  try {
    const ai = getClient();
    
    const context = metrics.mode === 'FILE' 
      ? `File Compression Operation. Original: ${(metrics.rawSize/1024).toFixed(2)} KB. Result: ${(metrics.processedSize/1024).toFixed(2)} KB. Ratio: ${metrics.compressionRatio?.toFixed(2)}%.` 
      : `Synthetic Matrix Benchmark. Size: ${metrics.sizeLabel}. Throughput: ${metrics.throughputMBps.toFixed(2)} MB/s.`;

    const prompt = `
      You are the AI Core of the Φ-Cloud Data Engine. 
      Analyze the completed operation.
      
      Operation Context:
      ${context}
      
      Performance Metrics:
      - Duration: ${metrics.duration.toFixed(4)}s
      - Speed: ${metrics.throughputMBps.toFixed(2)} MB/s
      
      Recent System Logs:
      ${logs.slice(-5).join('\n')}
      
      Provide a concise, technical summary of the operation's efficiency and success. 
      If it was a file compression, comment on the space saved.
      If it was a benchmark, comment on the system stability.
      Limit response to 2 sentences. Use professional, slightly futuristic engineering terminology.
    `;

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
    });

    return response.text || "Analysis complete. Data integrity verified.";
  } catch (error) {
    console.error("Gemini Analysis Failed:", error);
    return "CORE OFFLINE. UNABLE TO GENERATE ANALYSIS REPORT.";
  }
};