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