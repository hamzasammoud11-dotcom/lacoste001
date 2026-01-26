# BioFlow UI (Next.js)

This project is a migration of the Streamlit "old_ui" to Next.js 16, Shadcn UI, Framer Motion, and Tailwind CSS v4.

## Getting Started

1. **Install dependencies:**

    ```bash
    cd ui
    pnpm install
    # or
    npm install
    ```

2. **Run the development server:**

    ```bash
    pnpm dev
    # or
    npm run dev
    ```

3. Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Structure

- `app/`: App Router pages and layout.
  - `page.tsx`: Home Dashboard.
  - `discovery/`: Drug Discovery Pipeline interface.
  - `explorer/`: Data Explorer with Charts.
  - `data/`: Data Management.
  - `settings/`: Configuration.
  - `api/`: API Route handlers.
- `components/`: Reusable UI components.
  - `ui/`: Shadcn-like primitive components (Button, Card, etc.).
  - `sidebar.tsx`: Main Navigation.
  - `page-header.tsx`: Standard page headers.

## Tech Stack

- **Framework:** Next.js 16 (App Router)
- **Styling:** Tailwind CSS v4
- **UI Library:** Custom components inspired by Shadcn UI
- **Icons:** Lucide React
- **Charts:** Recharts
- **Animations:** Framer Motion (basic animations included, Framer Motion ready)

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
