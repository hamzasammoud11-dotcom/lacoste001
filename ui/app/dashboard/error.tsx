"use client";

import { useEffect } from "react";
import Link from "next/link";

import { RefreshCcw, Home, AlertCircle } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyTitle,
} from "@/components/ui/empty";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="flex w-full items-center justify-center p-4">
      <div className="flex min-h-screen items-center border-x">
        <div className="relative">
          <div className="absolute inset-x-0 top-0 h-px bg-border" />
          <Empty>
            <EmptyHeader>
              <div className="flex justify-center mb-4">
                <div className="p-4 rounded-full bg-destructive/10">
                  <AlertCircle className="h-12 w-12 text-destructive" />
                </div>
              </div>
              <EmptyTitle className="font-black font-mono text-5xl md:text-7xl">
                ERREUR
              </EmptyTitle>
              <EmptyDescription className="text-lg max-w-md mx-auto">
                Désolé, une erreur inattendue est survenue lors de la navigation.
                {error.digest && (
                  <span className="block mt-2 text-xs font-mono text-muted-foreground">
                    ID digest: {error.digest}
                  </span>
                )}
              </EmptyDescription>
            </EmptyHeader>
            <EmptyContent>
              <div className="flex flex-wrap items-center justify-center gap-4">
                <Button
                  onClick={() => reset()}
                  size="lg"
                  className="rounded-full px-8"
                >
                  <RefreshCcw className="mr-2 h-4 w-4" /> Réessayer
                </Button>
                <Button
                  asChild
                  variant="outline"
                  size="lg"
                  className="rounded-full px-8"
                >
                  <Link href="/">
                    <Home className="mr-2 h-4 w-4" /> Accueil
                  </Link>
                </Button>
              </div>
            </EmptyContent>
          </Empty>
          <div className="absolute inset-x-0 bottom-0 h-px bg-border" />
        </div>
      </div>
    </div>
  );
}
