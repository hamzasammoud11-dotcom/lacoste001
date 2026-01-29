import { Loader2 } from 'lucide-react';

export default function Loading() {
  return (
    <div className="flex h-[50vh] w-full items-center justify-center">
      <div className="flex flex-col items-center gap-2">
        <Loader2 className="text-primary h-8 w-8 animate-spin" />
        <p className="text-muted-foreground animate-pulse text-sm">
          Loading BioFlow...
        </p>
      </div>
    </div>
  );
}
