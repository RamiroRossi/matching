# POLICY IMPLICATIONS: ̨
# IMPROVEMENT_MATCHING_FUNCTION_level
# LOW Q CASE
using QuantEcon, Optim, Interpolations, LinearAlgebra,PlotlyJS,Distributions, Pkg
abstract type Matching
end

using QuantEcon, Optim, Interpolations, LinearAlgebra,PlotlyJS,Distributions, Pkg


struct dmp <: Matching
    pars::Dict{Symbol, Float64}
    #employment grid
    E::Vector{Float64}
    # agrid(jE)
    agrid::Vector{Float64}
    # αgrid(jE)
    αgrid::Vector{Float64}
    #wgrid(α,a)
    wgrid::Vector{Float64}
    #rVvgrid(α,a,w)
    rVv::Vector{Float64}
    #(α,a,w)
    rVu::Vector{Float64}
    # rVf
    rVf::Vector{Float64}
    #rVe
    rVe::Vector{Float64}
    # eq matrix
end



function dmp(;τw = 3, τu = 1, b=0.2,Cv = 1, 
    Cf = 1, A = 5, γ = 0.5, 
    κ = 1, β = 0.5, r = 0.02, NE = 10, se=0.001, L = NE + 1e-4)
    pars = Dict(:τw => τw, :τu => τu, 
    :b => b, :Cv=>Cv, :Cf => Cf, :A => A,
    :γ=> γ, :κ => κ, :β => β, :r => r, :NE => NE,:se =>se, :L=>L)
    E = collect(0.001:se:NE)
    agrid = similar(E)
    αgrid=similar(E)
    wgrid = similar(agrid)
    rVu = similar(agrid)
    rVv = similar(agrid)
    rVf = similar(agrid)
    rVe = similar(agrid)
    wgrid = similar(agrid)
    return dmp(pars, E, agrid, αgrid, wgrid, rVv, rVu,rVf,rVe)
end

include("C:/Users/rross/OneDrive/Documentos/julia/open_macro/Codigo-julia/dmp_functions.jl")

function q_iter!(τv,lab_descount,q,new_ne,iter)
    # this function applies the next algorithm
        # took a model 
        # calculate rvu at equilibrium
        # if value of being unemployed is above the home production productivity 
    model = dmp(;NE = new_ne, κ = τv, A = 10 ,τw = 7)
    qgmat = eq_values(model)
    ev = qgmat[1,5]
    a = av!(ev,model)
    α = αv!(ev,model)
    w = wv!(a,α,model)
    rvu_ = rvu!(a,w,model)
    rvu_eq = qgmat[1,2]
    if abs(rvu_ - rvu_eq) > 1e-2
        return print("RVU NO DA LO MISMO")
    end
    if rvu_ < q
        new_ne *= lab_descount
        iter += 1
        Lab = model.pars[:L]
        print("
        L iteration $iter, L = $Lab")
        if Lab < 0.01
            print("Labour market shrinked")
        end
        return q_iter!(τv,lab_descount,q,new_ne,iter)
    else 
        iter = 0
        ev = qgmat[1,5]
        αv = qgmat[1,6]
        av = qgmat[1,7]
        wv = qgmat[1,8]  
        # rvv
        rvv = qgmat[1,1]
        # rVu
        rvu = qgmat[1,2]
        # rVe 
        rve = qgmat[1,3]
        # rVf 
        rvf = qgmat[1,4]
        # L'
        lab_force = new_ne
        # self_employ 
        self_employ = 10 - new_ne
        #  unemployment
        uq = new_ne - ev
        # take home wage
        w_net = wv - model.pars[:τw]
        # drvu_dl
        index_eq = qgmat[1,10]
        return rvv, rvu, rve, rvf, ev, αv, av, wv, lab_force, self_employ, uq,w_net,index_eq
    end
end

model = dmp(; κ = 0.5, τw = 7, A = 10);
mat = eq_values(model)
mat[1,2]
mat[1,5]


function iter_loop(ft,lab_discount,q,tot_pop)
    τgrid = collect(0.5:0.01:ft)
    q_comparatives_x = zeros(15,size(τgrid,1))
    τ_iter = 0
    iter = 0
    Threads.@threads for jτ in eachindex(τgrid)
        iter = 0;
        τ_iter += 1;
        print("
        τ iterations $τ_iter
        ") 
        τv = τgrid[jτ]
        rvv, rvu, rve, rvf, ev, αv, av, wv, lab_force, self_employ, uq,w_net,index_eq= q_iter!(τv, lab_discount,q,tot_pop,iter)
        q_comparatives_x[1,jτ] = rvv
        q_comparatives_x[2,jτ] = rvu
        q_comparatives_x[3,jτ] = rve
        q_comparatives_x[4,jτ] = rvf
        q_comparatives_x[5,jτ] = ev
        q_comparatives_x[6,jτ] = αv 
        q_comparatives_x[7,jτ] = av 
        q_comparatives_x[8,jτ] = wv 
        q_comparatives_x[9,jτ] = uq
        q_comparatives_x[10,jτ] = lab_force
        q_comparatives_x[11,jτ] = self_employ
        q_comparatives_x[12,jτ] = w_net
        q_comparatives_x[14,jτ] = index_eq

    end
    return q_comparatives_x
end


κ_loop = iter_loop(5,0.99,1,10)

τgrid = collect(0.5:0.01:5)

# LEVEL

self_employment_low_tu = 10 .- (κ_loop[5,:] .+ κ_loop[9,:])
κ_loop[11,:]
employment_κ_low = plot([scatter(x= τgrid, y=κ_loop[5,:], 
name = "E*(κ)")], 
Layout(title="Employment level", xaxis_title="κ"));

unemployment_κ_low = plot([scatter(x= τgrid, y=κ_loop[9,:], 
    name = "U*(κ)")], 
Layout(title="Unemployment level", xaxis_title="κ"));

self_κ_low = plot([scatter(x= τgrid, y=self_employment_low_tu, 
name = "H*(κ)")], 
Layout(title="Self employment level", xaxis_title="κ"));

pop_plot_low_u = ([employment_κ_low unemployment_κ_low self_κ_low])
savefig(pop_plot_low_u,"replication_files/new_figures/improvement_matching/level.png")

# RATES

tot_pop_low_u = self_employment_low_tu .+ κ_loop[5,:] .+ κ_loop[9,:]
tot_pop_low_u = κ_loop[11,:] .+ κ_loop[5,:] .+ κ_loop[9,:]
e_rate_low = κ_loop[5,:] ./ tot_pop_low_u;
u_rate_low = κ_loop[9,:] ./ tot_pop_low_u;
self_rate_low = κ_loop[11,:] ./ tot_pop_low_u
plot_rates_low = plot([scatter(x=τgrid, 
y = e_rate_low, name="E/L"); scatter(x=τgrid, 
y=u_rate_low, name="U/L"); scatter(x=τgrid, 
y=self_rate_low, name="H/L")],
 Layout(title = "Employment and Unemployment rates, q = 1",xaxis_title = "κ"))

savefig(plot_rates_low, "replication_files/new_figures/improvement_matching/rates.png")
