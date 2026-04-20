import cyvcf2
import pandas as pd

def get_severity(mc_string):
    """Maps ClinVar MC terms to a numeric severity score (0-3)."""
    mc_string = str(mc_string).lower()
    if any(term in mc_string for term in ['stop_gained', 'frameshift', 'splice_acceptor', 'splice_donor']):
        return 3
    if 'missense' in mc_string or 'inframe' in mc_string:
        return 2
    if any(term in mc_string for term in ['synonymous', 'intron', '3_prime_utr', '5_prime_utr']):
        return 1
    return 0

def extract_variants(vcf_path: str) -> pd.DataFrame:
    print(f"Extracting high-signal genomic features...")
    vcf = cyvcf2.VCF(vcf_path)
    data = []

    for variant in vcf:
        clnsig = variant.INFO.get('CLNSIG')
        if not clnsig or 'Uncertain_significance' in clnsig:
            continue
            
        is_pathogenic = 1 if 'Pathogenic' in clnsig else 0

        freqs = [variant.INFO.get(f, 0.0) for f in ['AF_ESP', 'AF_EXAC', 'AF_TGP']]
        max_af = max([f if isinstance(f, float) else 0.0 for f in freqs])

        severity = get_severity(variant.INFO.get('MC', ""))

        data.append({
            'CHROM': variant.CHROM,
            'POS': variant.POS,
            'MAX_AF': max_af,
            'SEVERITY': severity,
            'QUAL': variant.QUAL if variant.QUAL is not None else 0.0,
            'LABEL': is_pathogenic
        })

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = extract_variants("data/raw/clinvar.vcf.gz") 
    df.to_csv("data/processed/variant_features.csv", index=False)
    print(f"Saved {len(df)} variants with Severity Scores.")