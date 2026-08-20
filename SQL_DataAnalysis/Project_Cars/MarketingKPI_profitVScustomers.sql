CREATE VIEW vpi_marketing_KPI
AS

WITH vip_table AS 
(SELECT o.customerNumber, SUM(quantityOrdered * (priceEach - buyPrice)) AS profit
  FROM products p
  JOIN orderdetails od
    ON p.productCode = od.productCode
  JOIN orders o
    ON o.orderNumber = od.orderNumber
 GROUP BY o.customerNumber
 ORDER BY profit)

SELECT contactLastName, contactFirstName, city, country, profit
FROM vip_table AS vt
JOIN customers AS c
ON vt.customerNumber = c.customerNumber
ORDER BY vt.profit DESC
LIMIT 5;

CREATE VIEW non_vip_marketing_KPI
AS

WITH non_vip_table AS 
(SELECT o.customerNumber, SUM(quantityOrdered * (priceEach - buyPrice)) AS profit
  FROM products p
  JOIN orderdetails od
    ON p.productCode = od.productCode
  JOIN orders o
    ON o.orderNumber = od.orderNumber
 GROUP BY o.customerNumber
 ORDER BY profit)

SELECT contactLastName, contactFirstName, city, country, profit
FROM non_vip_table AS nvt
JOIN customers AS c
ON nvt.customerNumber = c.customerNumber
ORDER BY nvt.profit ASC
LIMIT 5;

--SELECT *
--FROM vip_marketing_KPI
--LIMIT 10;

--SELECT *
--FROM non_vip_marketing_KPI
--LIMIT 10;