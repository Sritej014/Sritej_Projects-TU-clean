CREATE VIEW product_KPI
AS

WITH performance_table AS
(
    SELECT 
        p.productCode,
        SUM(o.quantityOrdered * o.priceEach) AS product_performance
    FROM products AS p
    JOIN orderdetails AS o
        ON p.productCode = o.productCode
    GROUP BY p.productCode
),
low_stock_table AS
(
    SELECT 
        p.productCode,
        ROUND(SUM(o.quantityOrdered) / p.quantityInStock / 1.0, 2) AS low_stock
    FROM products AS p
    JOIN orderdetails AS o
        ON p.productCode = o.productCode
    GROUP BY p.productCode
)

SELECT 
    pt.productCode,
    lt.low_stock,
    pt.product_performance
FROM performance_table AS pt
JOIN low_stock_table AS lt
    ON pt.productCode = lt.productCode
ORDER BY  pt.product_performance DESC
LIMIT 10;

--SELECT *
--FROM product_KPI
--LIMIT 10;
