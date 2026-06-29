"""Download confirmed transiting exoplanets from the NASA Exoplanet Archive.

Queries the TAP service for planets in the 'ps' table where 'tran_flag = 1'
and 'tic_id' is not null. Cleans the TIC IDs and writes them to
'confirmed_exoplanets.csv'.
"""

import urllib.request
import urllib.parse
import pandas as pd
import io
import sys

def download_confirmed_exoplanets(output_path='confirmed_exoplanets.csv'):
    print("Connecting to NASA Exoplanet Archive TAP service...")
    
    query = (
        "select distinct tic_id, pl_name, pl_orbper, pl_rade, pl_bmasse, st_teff "
        "from ps "
        "where tic_id is not null and tran_flag = 1"
    )
    
    params = {
        'query': query,
        'format': 'csv'
    }
    
    base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            data = response.read().decode('utf-8')
            
        print("Data downloaded successfully. Processing...")
        df = pd.read_csv(io.StringIO(data))
        
        df['tic_id'] = df['tic_id'].astype(str).str.replace('TIC', '', case=False).str.strip()
        
        df = df[df['tic_id'] != '']
        df = df[df['tic_id'].str.match(r'^\d+$', na=False)]
        
        df = df.drop_duplicates(subset=['pl_name'])
        
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} unique confirmed transiting planets to '{output_path}'.")
        print("\nFirst 5 entries:")
        print(df.head())
        
    except Exception as e:
        print(f"Error downloading confirmed exoplanets: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    download_confirmed_exoplanets()
"""
hello
aim for at least 10k to 20k samples generally.
"""