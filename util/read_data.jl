using CSV

function read_data(file, mat; delim='\t')
    d = CSV.read(file; delim=delim)
    for item in names(d)
           mat[2,parse(Int, string(item))+1,:] .= get.(d[:, item])
    end
end
