'use client';

import { useState, useEffect } from 'react';
import { ArrowLeft, Dna, ExternalLink, Loader2 } from 'lucide-react';
import Link from 'next/link';

export default function WorkflowPage() {
  const [isLoading, setIsLoading] = useState(true);
  const [langflowStatus, setLangflowStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  
  const LANGFLOW_URL = 'http://localhost:7860';
  
  useEffect(() => {
    const checkLangflow = async () => {
      try {
        const img = new Image();
        img.onload = () => setLangflowStatus('online');
        img.onerror = () => {
          fetch(`${LANGFLOW_URL}/api/v1/version`, { mode: 'no-cors' })
            .then(() => setLangflowStatus('online'))
            .catch(() => setLangflowStatus('offline'));
        };
        img.src = `${LANGFLOW_URL}/favicon.ico?t=${Date.now()}`;
        
        setTimeout(() => {
          if (langflowStatus === 'checking') {
            setLangflowStatus('online');
          }
        }, 2000);
      } catch {
        setLangflowStatus('offline');
      }
    };
    
    checkLangflow();
    
    const interval = setInterval(() => {
      if (langflowStatus === 'offline') {
        checkLangflow();
      }
    }, 5000);
    
    return () => clearInterval(interval);
  }, [langflowStatus]);
  
  const handleRetry = () => {
    setLangflowStatus('checking');
    setIsLoading(true);
  };
  
  // Clean white loading state
  if (langflowStatus === 'checking') {
    return (
      <div className="flex flex-col items-center justify-center h-full w-full bg-background">
        <Loader2 className="w-10 h-10 animate-spin text-primary mb-4" />
        <p className="text-muted-foreground">Connecting to Langflow...</p>
      </div>
    );
  }
  
  // Clean white offline state
  if (langflowStatus === 'offline') {
    return (
      <div className="flex flex-col h-full w-full bg-background">
        {/* Clean header */}
        <div className="flex items-center gap-4 px-4 py-3 border-b">
          <Link 
            href="/"
            className="flex items-center gap-2 text-muted-foreground"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to BioFlow
          </Link>
          <div className="flex items-center gap-2">
            <Dna className="w-5 h-5 text-primary" />
            <span className="font-semibold">BioFlow</span>
            <span className="text-muted-foreground">/</span>
            <span className="text-muted-foreground">Workflow Builder</span>
          </div>
        </div>

        {/* Offline message */}
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="max-w-lg text-center">
            <p className="text-muted-foreground mb-4">Langflow not running</p>
            <code className="block bg-muted text-foreground p-4 rounded-lg font-mono text-sm mb-4">
              langflow run --host 0.0.0.0 --port 7860
            </code>
            <p className="text-xs text-muted-foreground mb-4">
              Note: Run in a separate terminal/venv to avoid dependency conflicts
            </p>
            <button
              onClick={handleRetry}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-lg"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }
  
  // Clean white header with Langflow iframe
  return (
    <div className="h-full w-full flex flex-col bg-background">
      {/* Clean minimal header */}
      <div className="flex items-center justify-between px-4 py-2 border-b bg-background">
        <div className="flex items-center gap-4">
          <Link 
            href="/"
            className="flex items-center gap-2 text-muted-foreground"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to BioFlow
          </Link>
          
          <div className="h-6 w-px bg-border" />
          
          <div className="flex items-center gap-2">
            <Dna className="w-5 h-5 text-primary" />
            <span className="font-semibold">BioFlow</span>
            <span className="text-muted-foreground">/</span>
            <span className="text-muted-foreground">Workflow Builder</span>
          </div>
        </div>
        
        <a
          href={LANGFLOW_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 px-3 py-1.5 border hover:bg-muted rounded-lg text-sm transition-colors"
        >
          <ExternalLink className="w-4 h-4" />
          Open in New Tab
        </a>
      </div>
      
      {/* Full viewport Langflow iframe */}
      <div className="flex-1 relative">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-background z-10">
            <div className="flex flex-col items-center">
              <Loader2 className="w-10 h-10 animate-spin text-primary mb-3" />
              <p className="text-muted-foreground">Loading Langflow...</p>
            </div>
          </div>
        )}
        <iframe
          src={LANGFLOW_URL}
          className="w-full h-full border-0"
          onLoad={() => setIsLoading(false)}
          title="Langflow Workflow Builder"
          allow="clipboard-write; clipboard-read"
        />
      </div>
    </div>
  );
}
