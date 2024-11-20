import pandas as pd

def clean_fundamental_score(fundamental_score):
    columns_to_remove = ['ScoringId', 'ScoringType', 'Text']
    return fundamental_score.drop(columns=columns_to_remove)

def clean_rms_issuer(rms_issuer):
    columns_to_remove = ['PrimaryAnalystId', 'SecondaryAnalystId', 'ResearchTeam',
           'CompanyDescription', 'BondTicker',
           'OperatingCountryIso', 'Industry', 'Sponsor', 'MajorityOwnership',
           'MinorityOwnership', 'WhyInvested', 'CreditPositives',
           'CreditNegatives', 'CreditView', 'BookType', 
           'UpdateUser', 'SharePointExcelModel', 'SharePointSiteName',
           'SharePointProvisioningStatus', 'SubIndustry',
           'SharePointProvisioningMessage']  # 'Status',
    return rms_issuer.drop(columns=columns_to_remove)

def get_rms_with_fundamental_score(fundamental_score, rms_issuer):
    # Clean data
    fundamental_score = clean_fundamental_score(fundamental_score)
    rms_issuer = clean_rms_issuer(rms_issuer)
    # Merge
    rms_with_fundamental_score = fundamental_score.merge(rms_issuer, on='RmsId', how='left')
    rms_with_fundamental_score["SharePointLinkTruncated"] = rms_with_fundamental_score["SharePointLink"].apply(lambda x: x[:-1] if str(x).endswith('/') else x)
    return rms_with_fundamental_score

def process_sp_data(sp_data):
    sp_data_unique = sp_data.drop_duplicates(subset='EB_SPWebUrl')
    sp_data_unique['EB_SPWebUrl_cleaned'] = sp_data_unique['EB_SPWebUrl'].astype(str).str.rstrip('/')
    return sp_data_unique

def merge_data(sp_data_unique, rms_with_fundamental_score):
    rms_with_fundamental_score['SharePointLinkTruncated_cleaned'] = rms_with_fundamental_score['SharePointLinkTruncated'].astype(str).str.rstrip('/')
    merged_data = sp_data_unique.merge(
        rms_with_fundamental_score[['SharePointLinkTruncated_cleaned', 'RmsId', 'Status']],
        left_on='EB_SPWebUrl_cleaned',
        right_on='SharePointLinkTruncated_cleaned',
        how='left')
    merged_data = merged_data.dropna(subset=["RmsId"])
    merged_data["RmsId"] = merged_data["RmsId"].astype(int)
    merged_data_unique = merged_data.drop_duplicates(subset='EB_SPWebUrl_cleaned')
    return merged_data_unique