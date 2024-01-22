show databases;
use inventory_db;
show tables;

select * from pharmacy_tbl;

/* ################################# Bussiness Moment Decisions ########################################################################### */


(SELECT AVG(Quantity) AS mean_value, MIN(Quantity), MAX(Quantity),( MAX(Quantity)-MIN(Quantity)) AS RANGE_,
  SUM((Quantity- mean_value) * (Quantity - mean_value)) / COUNT(*) AS variance, 
  sqrt(SUM((Quantity- mean_value) * (Quantity - mean_value)) / COUNT(*)) as SD,
  
  
  (COUNT(Quantity) / ((COUNT(Quantity) - 1) * (COUNT(Quantity) - 2))) *
        SUM(POW(Quantity - (SELECT AVG(Quantity) FROM pharmacy_tbl), 3)) /
        POW(STDDEV(Quantity), 3) AS Skewness,
        
        (SUM(POW(Quantity - (SELECT AVG(Quantity) FROM pharmacy_tbl), 4)) / COUNT(Quantity)) / 
POW(STDDEV(Quantity), 4) - ((3 * POW((COUNT(Quantity) - 1), 2)) / ((COUNT(Quantity) - 2) * (COUNT(Quantity) - 3))) as Kurtosis
        
  FROM (
    -- Calculate the mean (average) of the column
    SELECT
        AVG(Quantity) AS mean_value
    FROM pharmacy_tbl
) as subquery, pharmacy_tbl);

(SELECT AVG(ReturnQuantity) AS mean_value, MIN(ReturnQuantity), MAX(ReturnQuantity),( MAX(ReturnQuantity)-MIN(ReturnQuantity)) AS RANGE_,
  SUM((ReturnQuantity - mean_value) * (ReturnQuantity - mean_value)) / COUNT(*) AS variance, 
  sqrt(SUM((ReturnQuantity- mean_value) * (ReturnQuantity- mean_value)) / COUNT(*)) as SD,
  
  
  
  (COUNT(ReturnQuantity) / ((COUNT(ReturnQuantity) - 1) * (COUNT(ReturnQuantity) - 2))) *
        SUM(POW(ReturnQuantity - (SELECT AVG(ReturnQuantity) FROM pharmacy_tbl), 3)) /
        POW(STDDEV(ReturnQuantity), 3) AS Skewness,
        
        (SUM(POW(ReturnQuantity - (SELECT AVG(ReturnQuantity) FROM pharmacy_tbl), 4)) / COUNT(ReturnQuantity)) / POW(STDDEV(ReturnQuantity), 4) - ((3 * POW((COUNT(ReturnQuantity) - 1), 2)) / ((COUNT(ReturnQuantity) - 2) * (COUNT(ReturnQuantity) - 3))) as Kurtosis
   
   
  FROM (
    -- Calculate the mean (average) of the column
    SELECT
        AVG(ReturnQuantity) AS mean_value
    FROM pharmacy_tbl
) as subquery, pharmacy_tbl);

(SELECT AVG(rtnmrp) AS mean_value, MIN(rtnmrp), MAX(rtnmrp),( MAX(rtnmrp)-MIN(rtnmrp)) AS RANGE_,
  SUM((rtnmrp - mean_value) * (rtnmrp - mean_value)) / COUNT(*) AS variance, 
  sqrt(SUM((rtnmrp - mean_value) * (rtnmrp - mean_value)) / COUNT(*)) as SD,
  
  
  (COUNT(rtnmrp) / ((COUNT(rtnmrp) - 1) * (COUNT(rtnmrp) - 2))) *
        SUM(POW(rtnmrp - (SELECT AVG(rtnmrp) FROM pharmacy_tbl), 3)) /
        POW(STDDEV(rtnmrp), 3) AS Skewness,
        
        (SUM(POW(rtnmrp - (SELECT AVG(rtnmrp) FROM pharmacy_tbl), 4)) / COUNT(rtnmrp)) / POW(STDDEV(rtnmrp), 4) - ((3 * POW((COUNT(rtnmrp) - 1), 2)) / ((COUNT(rtnmrp) - 2) * (COUNT(rtnmrp) - 3))) as Kurtosis
   
  
  
  FROM (
    -- Calculate the mean (average) of the column
    SELECT
        AVG(rtnmrp) AS mean_value
    FROM pharmacy_tbl
) as subquery, pharmacy_tbl);

(SELECT AVG(Final_Cost) AS mean_value, MIN(Final_Cost), MAX(Final_Cost),( MAX(Final_Cost)-MIN(Final_Cost)) AS RANGE_,
  SUM((Final_Cost - mean_value) * (Final_Cost - mean_value)) / COUNT(*) AS variance, 
  sqrt(SUM((Final_Cost - mean_value) * (Final_Cost - mean_value)) / COUNT(*)) as SD,
  
  
  (COUNT(Final_Cost) / ((COUNT(Final_Cost) - 1) * (COUNT(Final_Cost) - 2))) *
        SUM(POW(Final_Cost - (SELECT AVG(Final_Cost) FROM pharmacy_tbl), 3)) /
        POW(STDDEV(Final_Cost), 3) AS Skewness,
        
        (SUM(POW(Final_Cost - (SELECT AVG(Final_Cost) FROM pharmacy_tbl), 4)) / 
        COUNT(Final_Cost)) / POW(STDDEV(Final_Cost), 4) - ((3 * POW((COUNT(Final_Cost) - 1), 2)) /
        ((COUNT(Final_Cost) - 2) * (COUNT(Final_Cost) - 3))) as Kurtosis
   
  
  
  FROM (
    -- Calculate the mean (average) of the column
    SELECT
        AVG(rtnmrp) AS mean_value
    FROM pharmacy_tbl
) as subquery, pharmacy_tbl);


(SELECT AVG(Final_Sales) AS mean_value, MIN(Final_Sales), MAX(Final_Sales),( MAX(Final_Sales)-MIN(Final_Sales)) AS RANGE_,
  SUM((Final_Sales- mean_value) * (Final_Sales - mean_value)) / COUNT(*) AS variance, 
  sqrt(SUM((Final_Sales - mean_value) * (Final_Sales - mean_value)) / COUNT(*)) as SD,
  
  
  (COUNT(Final_Sales) / ((COUNT(Final_Sales) - 1) * (COUNT(Final_Sales) - 2))) *
        SUM(POW(Final_Sales - (SELECT AVG(Final_Sales) FROM pharmacy_tbl), 3)) /
        POW(STDDEV(Final_Sales), 3) AS Skewness,
        
        (SUM(POW(Final_Sales - (SELECT AVG(Final_Sales) FROM pharmacy_tbl), 4)) / 
        COUNT(Final_Sales)) / POW(STDDEV(Final_Sales), 4) - ((3 * POW((COUNT(Final_Sales) - 1), 2)) /
        ((COUNT(Final_Sales) - 2) * (COUNT(Final_Sales) - 3))) as Kurtosis
   
  
  
  FROM (
    -- Calculate the mean (average) of the column
    SELECT
        AVG(rtnmrp) AS mean_value
    FROM pharmacy_tbl
) as subquery, pharmacy_tbl);



############################### Value Counts and Most repeated value #########################################################
select SubCat1, count(*) as value_counts from pharmacy_tbl group by SubCat1 order by value_counts desc limit 1;
select SubCat, count(*) as value_counts from pharmacy_tbl group by SubCat order by value_counts desc limit 1;
select Specialisation, count(*) as value_counts from pharmacy_tbl group by Specialisation order by value_counts desc limit 1;
select Dept, count(*) as value_counts from pharmacy_tbl group by Dept order by value_counts desc limit 1;
select DrugName, count(*) as value_counts from pharmacy_tbl group by DrugName order by value_counts desc limit 1;
select Formulation, count(*) as value_counts from pharmacy_tbl group by Formulation order by value_counts desc limit 1;

/* ########################################## Missing_Values ######################################## */
select count(*) from pharmacy_tbl where DrugName is null or trim(DrugName) ='';
set @mode_ := (select DrugName from (select DrugName, count(*) as  dept_count from pharmacy_tbl group by DrugName order by dept_count desc limit 2) as subquery order by dept_count limit 1);
set SQL_SAFE_UPDATES = 0;
UPDATE pharmacy_tbl SET DrugName = @mode_ where DrugName is null or trim(DrugName) ='';

select count(*) from pharmacy_tbl where Formulation is null or trim(Formulation) ='';
set @mode_ := (select Formulation from (select Formulation, count(*) as  Formulation_count from pharmacy_tbl
 group by Formulation order by Formulation_count desc limit 1) as subquery limit 1);
set SQL_SAFE_UPDATES = 0;
UPDATE pharmacy_tbl SET Formulation = @mode_ where Formulation is null or trim(Formulation) ='';

select count(*) from pharmacy_tbl where SubCat is null or trim(SubCat) ='';
set @mode_ := (select SubCat from (select SubCat, count(*) as  SubCat_count from pharmacy_tbl group by SubCat order by SubCat_count desc limit 1) as subquery limit 1);
set SQL_SAFE_UPDATES = 0;
UPDATE pharmacy_tbl SET SubCat = @mode_ where SubCat is null or trim(SubCat) ='';

select count(*) from pharmacy_tbl where SubCat1 is null or trim(SubCat1) ='';
set @mode_ := (select SubCat1 from (select SubCat1, count(*) as  SubCat1_count from pharmacy_tbl group by SubCat1 order by SubCat1_count desc limit 1) as subquery limit 1);
set SQL_SAFE_UPDATES = 0;
UPDATE pharmacy_tbl SET SubCat1 = @mode_ where SubCat1 is null or trim(SubCat1) ='';
/* ###################################################################################################################################*/


/* ################################### Outliers Analysis ########################################################################### */
SELECT @LowerBound :=  (avg(Quantity)- 3*stddev(Quantity))  , @UpperBound := (avg(Quantity) + 3*stddev(Quantity)) FROM pharmacy_tbl;
select count(*) from pharmacy_tbl where Quantity < @LowerBound or Quantity > @UpperBound;
-- Detect and replace outliers with minimum and maximum values
UPDATE pharmacy_tbl SET Quantity = CASE WHEN Quantity < @LowerBound THEN @LowerBound WHEN Quantity > @UpperBound THEN @UpperBound ELSE Quantity END;






SELECT @LowerBound :=  (avg(ReturnQuantity)- 3*stddev(ReturnQuantity))  , @UpperBound := (avg(ReturnQuantity) + 3*stddev(ReturnQuantity)) FROM pharmacy_tbl;
select count(*) from pharmacy_tbl where ReturnQuantity < @LowerBound or ReturnQuantity > @UpperBound;
-- Detect and replace outliers with minimum and maximum values
UPDATE pharmacy_tbl SET ReturnQuantity = CASE WHEN ReturnQuantity < @LowerBound THEN @LowerBound WHEN ReturnQuantity > @UpperBound THEN @UpperBound ELSE ReturnQuantity END;






SELECT @LowerBound :=  (avg(Final_Cost)- 3*stddev(Final_Cost))  , @UpperBound := (avg(Final_Cost) + 3*stddev(Final_Cost)) FROM pharmacy_tbl;
select count(*) from pharmacy_tbl where Final_Cost < @LowerBound or Final_Cost > @UpperBound;
-- Detect and replace outliers with minimum and maximum values
UPDATE pharmacy_tbl SET Final_Cost = CASE WHEN Final_Cost < @LowerBound THEN @LowerBound WHEN Final_Cost > @UpperBound THEN @UpperBound ELSE Final_Cost END;




SELECT @LowerBound :=  (avg(Final_Sales)- 3*stddev(Final_Sales))  , @UpperBound := (avg(Final_Sales) + 3*stddev(Final_Sales)) FROM pharmacy_tbl;
select count(*) from pharmacy_tbl where Final_Sales < @LowerBound or Final_Sales > @UpperBound;
-- Detect and replace outliers with minimum and maximum values
UPDATE pharmacy_tbl SET Final_Sales = CASE WHEN Final_Sales < @LowerBound THEN @LowerBound WHEN Final_Sales > @UpperBound THEN @UpperBound ELSE Final_Sales END;




SELECT @LowerBound :=  (avg(rtnmrp)- 3*stddev(rtnmrp))  , @UpperBound := (avg(rtnmrp) + 3*stddev(rtnmrp)) FROM pharmacy_tbl;
select count(*) from pharmacy_tbl where rtnmrp < @LowerBound or rtnmrp > @UpperBound;
-- Detect and replace outliers with minimum and maximum values
UPDATE pharmacy_tbl SET rtnmrp = CASE WHEN rtnmrp < @LowerBound THEN @LowerBound WHEN rtnmrp > @UpperBound THEN @UpperBound ELSE rtnmrp END;

#########################################################
            
/* Handling Duplicates */
select Typeofsales,Patient_ID,Specialisation,Dept,Dateofbill,Quantity,ReturnQuantity,Final_Cost,Final_Sales,RtnMRP,Formulation,DrugName,SubCat,SubCat1,count(*) as count from med_inv
group by Typeofsales,Patient_ID,Specialisation,Dept,Dateofbill,Quantity,ReturnQuantity,Final_Cost,Final_Sales,RtnMRP,Formulation,DrugName,SubCat,SubCat1
having count > 1;

delete from med_inv
where (Patient_ID, Dateofbill, DrugName) in (
select t.Patient_ID, t.Dateofbill, t.DrugName
from (
select Patient_ID, Dateofbill, DrugName
from med_inv
group by Patient_ID, Dateofbill, DrugName
having COUNT(*) > 1
) as t
);
select count(*) as total_rows from med_inv; /*13923*/

#########################################################




/* ################################# Bussiness Moment Decisions after preprocessing ########################################################################### */


(SELECT AVG(Quantity) AS mean_value, MIN(Quantity), MAX(Quantity),( MAX(Quantity)-MIN(Quantity)) AS RANGE_,
  SUM((Quantity- mean_value) * (Quantity - mean_value)) / COUNT(*) AS variance, 
  sqrt(SUM((Quantity- mean_value) * (Quantity - mean_value)) / COUNT(*)) as SD,
  
  
  (COUNT(Quantity) / ((COUNT(Quantity) - 1) * (COUNT(Quantity) - 2))) *
        SUM(POW(Quantity - (SELECT AVG(Quantity) FROM pharmacy_tbl), 3)) /
        POW(STDDEV(Quantity), 3) AS Skewness,
        
        (SUM(POW(Quantity - (SELECT AVG(Quantity) FROM pharmacy_tbl), 4)) / COUNT(Quantity)) / POW(STDDEV(Quantity), 4) - ((3 * POW((COUNT(Quantity) - 1), 2)) / ((COUNT(Quantity) - 2) * (COUNT(Quantity) - 3))) as Kurtosis
        
  FROM (
    -- Calculate the mean (average) of the column
    SELECT
        AVG(Quantity) AS mean_value
    FROM pharmacy_tbl
) as subquery, pharmacy_tbl);

(SELECT AVG(ReturnQuantity) AS mean_value, MIN(ReturnQuantity), MAX(ReturnQuantity),( MAX(ReturnQuantity)-MIN(ReturnQuantity)) AS RANGE_,
  SUM((ReturnQuantity - mean_value) * (ReturnQuantity - mean_value)) / COUNT(*) AS variance, 
  sqrt(SUM((ReturnQuantity- mean_value) * (ReturnQuantity- mean_value)) / COUNT(*)) as SD,
  
  
  
  (COUNT(ReturnQuantity) / ((COUNT(ReturnQuantity) - 1) * (COUNT(ReturnQuantity) - 2))) *
        SUM(POW(ReturnQuantity - (SELECT AVG(ReturnQuantity) FROM pharmacy_tbl), 3)) /
        POW(STDDEV(ReturnQuantity), 3) AS Skewness,
        
        (SUM(POW(ReturnQuantity - (SELECT AVG(ReturnQuantity) FROM pharmacy_tbl), 4)) / COUNT(ReturnQuantity)) / POW(STDDEV(ReturnQuantity), 4) - ((3 * POW((COUNT(ReturnQuantity) - 1), 2)) / ((COUNT(ReturnQuantity) - 2) * (COUNT(ReturnQuantity) - 3))) as Kurtosis
   
   
  FROM (
    -- Calculate the mean (average) of the column
    SELECT
        AVG(ReturnQuantity) AS mean_value
    FROM pharmacy_tbl
) as subquery, pharmacy_tbl);

(SELECT AVG(rtnmrp) AS mean_value, MIN(rtnmrp), MAX(rtnmrp),( MAX(rtnmrp)-MIN(rtnmrp)) AS RANGE_,
  SUM((rtnmrp - mean_value) * (rtnmrp - mean_value)) / COUNT(*) AS variance, 
  sqrt(SUM((rtnmrp - mean_value) * (rtnmrp - mean_value)) / COUNT(*)) as SD,
  
  
  (COUNT(rtnmrp) / ((COUNT(rtnmrp) - 1) * (COUNT(rtnmrp) - 2))) *
        SUM(POW(rtnmrp - (SELECT AVG(rtnmrp) FROM pharmacy_tbl), 3)) /
        POW(STDDEV(rtnmrp), 3) AS Skewness,
        
        (SUM(POW(rtnmrp - (SELECT AVG(rtnmrp) FROM pharmacy_tbl), 4)) / COUNT(rtnmrp)) / POW(STDDEV(rtnmrp), 4) - ((3 * POW((COUNT(rtnmrp) - 1), 2)) / ((COUNT(rtnmrp) - 2) * (COUNT(rtnmrp) - 3))) as Kurtosis
   
  
  
  FROM (
    -- Calculate the mean (average) of the column
    SELECT
        AVG(rtnmrp) AS mean_value
    FROM pharmacy_tbl
) as subquery, pharmacy_tbl);

(SELECT AVG(Final_Cost) AS mean_value, MIN(Final_Cost), MAX(Final_Cost),( MAX(Final_Cost)-MIN(Final_Cost)) AS RANGE_,
  SUM((Final_Cost - mean_value) * (Final_Cost - mean_value)) / COUNT(*) AS variance, 
  sqrt(SUM((Final_Cost - mean_value) * (Final_Cost - mean_value)) / COUNT(*)) as SD,
  
  
  (COUNT(Final_Cost) / ((COUNT(Final_Cost) - 1) * (COUNT(Final_Cost) - 2))) *
        SUM(POW(Final_Cost - (SELECT AVG(Final_Cost) FROM pharmacy_tbl), 3)) /
        POW(STDDEV(Final_Cost), 3) AS Skewness,
        
        (SUM(POW(Final_Cost - (SELECT AVG(Final_Cost) FROM pharmacy_tbl), 4)) / 
        COUNT(Final_Cost)) / POW(STDDEV(Final_Cost), 4) - ((3 * POW((COUNT(Final_Cost) - 1), 2)) /
        ((COUNT(Final_Cost) - 2) * (COUNT(Final_Cost) - 3))) as Kurtosis
   
  
  
  FROM (
    -- Calculate the mean (average) of the column
    SELECT
        AVG(rtnmrp) AS mean_value
    FROM pharmacy_tbl
) as subquery, pharmacy_tbl);


(SELECT AVG(Final_Sales) AS mean_value, MIN(Final_Sales), MAX(Final_Sales),( MAX(Final_Sales)-MIN(Final_Sales)) AS RANGE_,
  SUM((Final_Sales- mean_value) * (Final_Sales - mean_value)) / COUNT(*) AS variance, 
  sqrt(SUM((Final_Sales - mean_value) * (Final_Sales - mean_value)) / COUNT(*)) as SD,
  
  
  (COUNT(Final_Sales) / ((COUNT(Final_Sales) - 1) * (COUNT(Final_Sales) - 2))) *
        SUM(POW(Final_Sales - (SELECT AVG(Final_Sales) FROM pharmacy_tbl), 3)) /
        POW(STDDEV(Final_Sales), 3) AS Skewness,
        
        (SUM(POW(Final_Sales - (SELECT AVG(Final_Sales) FROM pharmacy_tbl), 4)) / 
        COUNT(Final_Sales)) / POW(STDDEV(Final_Sales), 4) - ((3 * POW((COUNT(Final_Sales) - 1), 2)) /
        ((COUNT(Final_Sales) - 2) * (COUNT(Final_Sales) - 3))) as Kurtosis
   
  
  
  FROM (
    -- Calculate the mean (average) of the column
    SELECT
        AVG(rtnmrp) AS mean_value
    FROM pharmacy_tbl
) as subquery, pharmacy_tbl);



############################### Value Counts and Most repeated value #########################################################
select SubCat1, count(*) as value_counts from pharmacy_tbl group by SubCat1 order by value_counts desc limit 1;
select SubCat, count(*) as value_counts from pharmacy_tbl group by SubCat order by value_counts desc limit 1;
select Specialisation, count(*) as value_counts from pharmacy_tbl group by Specialisation order by value_counts desc limit 1;
select Dept, count(*) as value_counts from pharmacy_tbl group by Dept order by value_counts desc limit 1;
select DrugName, count(*) as value_counts from pharmacy_tbl group by DrugName order by value_counts desc limit 1;
select Formulation, count(*) as value_counts from pharmacy_tbl group by Formulation order by value_counts desc limit 1;
