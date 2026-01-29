'use client';

import * as React from 'react';

interface ProjectState {
  currentProject: string | null;
  setCurrentProject: (project: string | null) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
}

const ProjectStateContext = React.createContext<ProjectState | undefined>(
  undefined,
);

export function ProjectStateProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [currentProject, setCurrentProject] = React.useState<string | null>(
    null,
  );
  const [isLoading, setIsLoading] = React.useState(false);

  const value = React.useMemo(
    () => ({
      currentProject,
      setCurrentProject,
      isLoading,
      setIsLoading,
    }),
    [currentProject, isLoading],
  );

  return (
    <ProjectStateContext.Provider value={value}>
      {children}
    </ProjectStateContext.Provider>
  );
}

export function useProjectState() {
  const context = React.useContext(ProjectStateContext);
  if (context === undefined) {
    throw new Error(
      'useProjectState must be used within a ProjectStateProvider',
    );
  }
  return context;
}
