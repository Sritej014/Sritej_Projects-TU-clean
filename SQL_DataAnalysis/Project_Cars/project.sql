SELECT 'customers' AS table_name,
    (SELECT COUNT(*) FROM pragma_table_info('customers')) AS number_of_attributes,
    COUNT(*) AS number_of_rows
FROM customers


UNION ALL

SELECT 'products' AS table_name,
    (SELECT COUNT(*) FROM pragma_table_info('products')) AS number_of_attributes,
    COUNT(*) AS number_of_rows
FROM products

UNION ALL

SELECT 'productlines' AS table_name,
    (SELECT COUNT(*) FROM pragma_table_info('productlines')) AS number_of_attributes,
    COUNT(*) AS number_of_rows
FROM productlines

UNION ALL

SELECT 'orders' AS table_name,
    (SELECT COUNT(*) FROM pragma_table_info('orders')) AS number_of_attributes,
    COUNT(*) AS number_of_rows
FROM orders

UNION ALL

SELECT 'orderdetails' AS table_name,
    (SELECT COUNT(*) FROM pragma_table_info('orderdetails')) AS number_of_attributes,
    COUNT(*) AS number_of_rows
FROM orderdetails

UNION ALL

SELECT 'payments' AS table_name,
    (SELECT COUNT(*) FROM pragma_table_info('payments')) AS number_of_attributes,
    COUNT(*) AS number_of_rows
FROM payments

UNION ALL

SELECT 'employees' AS table_name,
    (SELECT COUNT(*) FROM pragma_table_info('employees')) AS number_of_attributes,
    COUNT(*) AS number_of_rows
FROM employees

UNION ALL

SELECT 'offices' AS table_name,
    (SELECT COUNT(*) FROM pragma_table_info('offices')) AS number_of_attributes,
    COUNT(*) AS number_of_rows
FROM offices