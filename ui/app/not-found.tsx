import { Compass, Home } from 'lucide-react';
import Link from 'next/link';

import { Button } from '@/components/ui/button';
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyTitle,
} from '@/components/ui/empty';

export default function NotFound() {
  return (
    <div className="flex w-full items-center justify-center">
      <div className="flex h-screen items-center border-x">
        <div className="relative">
          <div className="bg-border absolute inset-x-0 top-0 h-px" />
          <Empty>
            <EmptyHeader>
              <EmptyTitle className="font-mono text-8xl font-black md:text-9xl">
                404
              </EmptyTitle>
              <EmptyDescription className="text-lg text-nowrap">
                La page que vous recherchez a peut-être été
                <br />
                déplacée ou n&apos;existe plus.
              </EmptyDescription>
            </EmptyHeader>
            <EmptyContent>
              <div className="flex gap-4">
                <Button asChild size="lg" className="rounded-full px-8">
                  <Link href="/">
                    <Home className="mr-2 h-4 w-4" /> Accueil
                  </Link>
                </Button>
                <Button
                  asChild
                  variant="outline"
                  size="lg"
                  className="rounded-full px-8"
                >
                  <Link href="/about">
                    <Compass className="mr-2 h-4 w-4" /> Explorer
                  </Link>
                </Button>
              </div>
            </EmptyContent>
          </Empty>
          <div className="bg-border absolute inset-x-0 bottom-0 h-px" />
        </div>
      </div>
    </div>
  );
}
