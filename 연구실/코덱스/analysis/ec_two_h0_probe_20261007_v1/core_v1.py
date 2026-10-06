DROP=('in_co2_h0','act_heating_h0')
def project_columns(columns):
    cols=list(columns)
    if len(cols)!=len(set(cols)) or not set(DROP).issubset(cols):
        raise ValueError('Expected unique columns containing both fixed h0 features')
    return [c for c in cols if c not in DROP]
