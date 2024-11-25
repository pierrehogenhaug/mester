# src/data_processing/database_utils.py

from capfourpy.databases import Database
import pandas as pd


def get_fundamental_score(db):
    sql_query = """
    WITH tbl1 AS(
    	SELECT r.ScoringId,
    		   r.RmsId,
    		   t.TemplateName AS ScoringType,
    		   r.ScoringDate,
    		   cat.Grouping AS CategoryGroup,
    		   cat.Name AS Category,
    		   rc.Score,
    		   rc.Text,
    		   (
    			   SELECT c.Description AS CharacteristicText,
    					  c.Influence AS CharacteristicInfluence
    			   FROM Scoring.ResultCharacteristic AS rca
    				   LEFT JOIN Scoring.Characteristic AS c ON c.CategoryId = rca.CategoryId AND c.CharacteristicId = rca.CharacteristicId
    			   WHERE rca.ScoringId = rc.ScoringId AND rca.CategoryId = rc.CategoryId
    			   FOR JSON PATH
    		   ) AS TaggedCharacteristics
    	FROM Scoring.Result AS r
    		INNER JOIN Scoring.Template AS t ON t.TemplateId = r.TemplateId
    		INNER JOIN Scoring.ResultCategory AS rc ON rc.ScoringId = r.ScoringId
    		INNER JOIN Scoring.Category AS cat ON cat.CategoryId = rc.CategoryId
    	WHERE t.TemplateName = 'Corporate'
    )
    SELECT * FROM tbl1 WHERE TaggedCharacteristics IS NOT NULL
    """
    return db.read_sql(sql_query)


def get_rms_issuer(db):
    sql_query = """
    SELECT *
    FROM [CfRms_prod].[Core].[RmsIssuer]
    WHERE SharePointLink IS NOT NULL
    """
    return db.read_sql(sql_query)


def get_isin_rms_link(db):
	sql_query = """
	SELECT
		i.RmsId,
		a.AssetId,
		a.IssuerId,
		a.ISIN,
		a.IssuerName,
		a.AssetName,
		a.IssueDate,
		i.AbbrevName
	FROM
		C4DW.DailyOverview.AssetData a
	LEFT JOIN
		C4DW.DailyOverview.IssuerData i
	ON
		a.IssuerId = i.IssuerId
	WHERE
		a.AssetType = 'Bond'
		AND a.OperatingCountryISO NOT LIKE 'US'
		AND a.ISIN IS NOT NULL
		AND i.RmsId IS NOT NULL
	"""
	return db.read_sql(sql_query)


def get_findox_mapping_with_rms(db):
	sql_query = """
	-- Get Identifiers from Findox.Identifier as Columns
	WITH FindoxIdentifiers AS (
		SELECT 
			IssuerId,
			MAX(CASE WHEN IdentifierTypeName = 'Isin' THEN Identifier END) AS Isin,
			MAX(CASE WHEN IdentifierTypeName = 'Lxid' THEN Identifier END) AS Lxid,
			MAX(CASE WHEN IdentifierTypeName = 'Figi' THEN Identifier END) AS Figi,
			MAX(CASE WHEN IdentifierTypeName = 'BloombergId' THEN Identifier END) AS BloombergId
		FROM 
			CfAnalytics.Findox.Identifier
		GROUP BY 
			IssuerId
	),

	-- Select Asset Data from DailyOverview.AssetData
	AssetData AS (
		SELECT 
			IssuerId,
			PrimaryIdentifier,
			ISIN,
			Figi,
			BloombergID,
			BloombergUniqueID,
			LoanXID,
			BondTicker
		FROM 
			[C4DW].[DailyOverview].[AssetData]
	),

	-- Filter Issuer Mappings for 'Everest' Type
	EverestIssuerMapping AS (
		SELECT 
			*
		FROM 
			CfAnalytics.Findox.IssuerMapping
		WHERE 
			ExtIssuerType = 'Everest'
	),

	IssuerData AS (
		SELECT 
			AbbrevName, 
			IssuerId,
			RmsId
		FROM
			[C4DW].[DailyOverview].[IssuerData]
	),

	-- Join FindoxIdentifiers with AssetData
	JoinedData AS (
		SELECT DISTINCT
			Asset.IssuerId AS EverestIssuerId,
			Findox.IssuerId AS FinDoxIssuerId,
			Asset.Isin AS EverestIsin,
			Findox.Isin AS FinDoxIsin,
			Asset.LoanXID AS EverestLoanXID,
			Findox.Lxid AS FinDoxLxid,
			Asset.Figi AS EverestFigi,
			Findox.Figi AS FinDoxFigi,
			Asset.BloombergId AS EverestBloombergId,
			Findox.BloombergId AS FinDoxBloombergId,
			ID.RmsId,
			ID.AbbrevName
		FROM 
			FindoxIdentifiers Findox
		LEFT JOIN 
			AssetData Asset
			ON Findox.Isin = Asset.ISIN 
			OR Findox.Isin = Asset.PrimaryIdentifier
			OR Findox.Lxid = Asset.LoanXID 
			OR Findox.Figi = Asset.Figi 
			OR Findox.BloombergId = Asset.BloombergID
			OR Findox.BloombergId = Asset.BloombergUniqueID
		LEFT JOIN 
			EverestIssuerMapping EIM
			ON Findox.IssuerId = EIM.FindoxIssuerId
		LEFT JOIN
			IssuerData ID
			ON Asset.IssuerId = ID.IssuerId
		WHERE 
			1=1
			--AND EIM.FindoxIssuerId IS NULL
			AND Asset.IssuerId IS NOT NULL
	)

	-- Get Distinct Everest and FinDox Issuer IDs
	SELECT DISTINCT 
		EverestIssuerId AS ExtIssuerId, 
		FinDoxIssuerId,
		AbbrevName,
		RmsId
	FROM 
		JoinedData
	WHERE 
      	RmsId IS NOT NULL
	"""
	return db.read_sql(sql_query)
