WITH vip_table AS 
(SELECT o.customerNumber, SUM(quantityOrdered * (priceEach - buyPrice)) AS profit, SUM(quantityOrdered) AS qO
  FROM products p
  JOIN orderdetails od
    ON p.productCode = od.productCode
  JOIN orders o
    ON o.orderNumber = od.orderNumber
 GROUP BY o.customerNumber
 ORDER BY profit)

SELECT c.contactLastName, c.contactFirstName, c.city, c.country, vt.profit/vt.qO AS avg_profit
FROM vip_table AS vt
JOIN customers AS c
ON vt.customerNumber = c.customerNumber
ORDER BY vt.profit DESC;