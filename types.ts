export enum SystemState {
  IDLE = 'IDLE',
  INITIALIZING = 'INITIALIZING',
  PROCESSING = 'PROCESSING',
  FINALIZING = 'FINALIZING',
  ANALYSIS = 'ANALYSIS',
  COMPLETED = 'COMPLETED',
  ERROR = 'ERROR',
  LISTENING = 'LISTENING' // New state for Network Mode
}

export interface SimulationLog {
  id: string;
  timestamp: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error' | 'system' | 'network';
}

export interface OperationMetrics {
  mode: 'BENCHMARK' | 'FILE' | 'API';
  sizeLabel: string; // "4096 x 4096" or "14.5 MB"
  rawSize: number;
  processedSize: number;
  throughputMBps: number;
  duration: number;
  compressionRatio?: number;
  originIp?: string; // For API requests
}

export interface FileData {
  file: File;
  preview?: string;
}

export interface NodeConfig {
  nodeId: string;
  port: number;
  endpoint: string;
  active: boolean;
}