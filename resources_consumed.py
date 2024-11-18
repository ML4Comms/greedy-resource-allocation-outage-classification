def calculate_res(cdf_array, R):
    resources_consumed = []  

    for cdf in cdf_array:  
        numerator = (1 - R * ((1 - cdf) ** (R - 1)) + (R - 1) * ((1 - cdf) ** R))
        denominator = cdf
        res = (numerator / denominator) + R * ((1 - cdf) ** (R - 1))
        
        resources_consumed.append(res)  

    return resources_consumed

# Example usage
cdf_array = [0.114767857,0.322632813,0.46289881,0.536325,0.659255208,0.718994792,0.766855469]  # Example array of cdf(q_{th}) values
R = 8  # Example value for |R|

res_values = calculate_res(cdf_array, R)
for cdf, res_value in zip(cdf_array, res_values):
    print(f"resources consumed = {res_value}")

