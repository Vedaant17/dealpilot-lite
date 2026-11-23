export interface CompanyOption {
  id: string;
  name: string;
}

// These IDs are the verified synthetic company identifiers in the backend seed data.
export const COMPANY_OPTIONS: CompanyOption[] = [
  { id: "aurora-retail", name: "Aurora Retail Solutions" },
  { id: "borealis-industrial", name: "Borealis Industrial Components" },
  { id: "cedar-health", name: "Cedar Health Analytics" },
];