# POLICY IMPLICATIONS: BIG

# A = 20, q = 1


using QuantEcon, Optim, Interpolations, LinearAlgebra,PlotlyJS,Distributions, Pkg, Roots
abstract type Matching
end

# DMP functions
function av!(ev,dmp::Matching)
    b,L = dmp.pars[:b],dmp.pars[:L]
    av = (b * ev)/(L - ev)
    # a = b*E/U
    return av
end

function αv!(ev,dmp::Matching)
    κ,γ,b,β,L  = dmp.pars[:κ],dmp.pars[:γ],dmp.pars[:b],dmp.pars[:β],dmp.pars[:L]
    αv = κ^(1/γ) * ((b*ev)^(1-(1/γ))) * ((L - ev)^(β/γ))  
    return αv
end

function wv!(ev,av,αv,dmp::Matching)
    r,b,Cf,Cv,τw,τu,A,B,L = dmp.pars[:r],dmp.pars[:b],dmp.pars[:Cf],dmp.pars[:Cv],dmp.pars[:τw],dmp.pars[:τu],dmp.pars[:A],dmp.pars[:B],dmp.pars[:L]
    wv = ((r+b+ av) * (A - Cf + Cv - B * L /ev) + (r + b + αv)*(τw - τu))/(2*r + 2*b + av + αv)
    return wv
end

function rvv!(ev,αv,wv,dmp::Matching)
    Cv,A,Cf,r,b,B,L = dmp.pars[:Cv],dmp.pars[:A],dmp.pars[:Cf],dmp.pars[:r],dmp.pars[:b],dmp.pars[:B],dmp.pars[:L]
    rvv = - Cv + αv * (A - wv - Cf + Cv - B*L/ev)/(r + b +αv) 
    return rvv
end

function rvu!(av,wv,dmp::Matching)
    τu,τw,r,b,B = dmp.pars[:τu],dmp.pars[:τw],dmp.pars[:r],dmp.pars[:b],dmp.pars[:B]
    rvu = - τu + av * (wv - τw + τu)/(r + b + av) + B 
    return rvu
end

function rvf!(ev,αv,wv,dmp::Matching)
    A,Cf,b,Cv,r,B,L = dmp.pars[:A],dmp.pars[:Cf],dmp.pars[:b],dmp.pars[:Cv],dmp.pars[:r],dmp.pars[:B],dmp.pars[:L]
    rvf = A - wv - Cf - b*((A - wv - Cf - B*L/ev + Cv)/(r + b + αv)) - B*L/ev
    return rvf
end

function rve!(av,wv,dmp::Matching)
    τw,τu,b,r,B,L = dmp.pars[:τw],dmp.pars[:τu],dmp.pars[:b],dmp.pars[:r],dmp.pars[:B],dmp.pars[:L]
   rve = wv - τw - b*((wv + τu - τw)/(av + b + r)) + B
   return rve 
end


function rV!(dmp::Matching) 
    for je in eachindex(dmp.E)
        # fill the vectors
        ev = dmp.E[je] 
        av = av!(ev,dmp)
        αv = αv!(ev,dmp)
        wv = wv!(ev,av,αv,dmp)
        # Value functions
        rvv = rvv!(ev,αv,wv,dmp)
        rvf = rvf!(ev,αv,wv,dmp)
        rve = rve!(av,wv,dmp)
        rvu = rvu!(av,wv,dmp)
        # Complete the vectors for each je
        dmp.rVv[je] = rvv
        dmp.rVu[je] = rvu
        dmp.rVf[je] = rvf
        dmp.rVe[je] = rve
        dmp.wgrid[je] = wv
        dmp.agrid[je] = av
        dmp.αgrid[je] = αv
    end
end


#########################
# EQUILIBRIUM SELECTION##
#########################
function rvv_zeros(dmp)
    # this function takes rvv as a function of E and parameters
    r,b,Cf,Cv,τw,τu,A,γ,β,L,κ,B = dmp.pars[:r],dmp.pars[:b],dmp.pars[:Cf],dmp.pars[:Cv],dmp.pars[:τw],dmp.pars[:τu],dmp.pars[:A],dmp.pars[:γ],dmp.pars[:β],dmp.pars[:L],dmp.pars[:κ],dmp.pars[:B]
    function f(ev)
    - Cv + (κ^(1/γ) * ((b*ev)^(1-(1/γ))) * ((L - ev)^(β/γ))) * (A - (((r+b+ ((b * ev)/(L - ev))) * (A - Cf + Cv - B * L /ev) + (r + b + (κ^(1/γ) * ((b*ev)^(1-(1/γ))) * ((L - ev)^(β/γ)) ))*(τw - τu))/(2*r + 2*b + ((b * ev)/(L - ev)) + (κ^(1/γ) * ((b*ev)^(1-(1/γ))) * ((L - ev)^(β/γ)) ))) - Cf + Cv - B*L/ev)/(r + b +(κ^(1/γ) * ((b*ev)^(1-(1/γ))) * ((L - ev)^(β/γ))))
end
zeros = find_zeros(f,minimum(dmp.E),dmp.pars[:L])
    return zeros
end

function zeros_elast(zeros,dmp)
    # calculate the zeros
    mat =ones(length(zeros),2)
    mat[:,1] = zeros[:,1]
    # for each zero I will calculate the derivatives
    for jz in eachindex(zeros)
        zv = zeros[jz]
        mat[jz,2] = rvvp!(zv,0.0001,dmp)
    end
    elast =mat[:,2]
    return elast
end

function selection!(dmp)
    # this function returns an employment in equilibrium once some conditions are satisfied
    # i) if the rvv is entirely < 0 then the eq is 0
    # ii) if it is positive look for the zeros, and return a positive value only if it is unique and got rvv'e* <0 
    # iii) if it is unique and rvv'e > 0 return 0
    # iv) if it is not unique take the equilibriums with negative rvv'e*. return the maximum
    # check if the vector is positive (check that it is not negative)
    rvv_posit = filter(x -> x>=0,dmp.rVv)
    if rvv_posit != []
        # Find the equilibrium
        zeros = rvv_zeros(dmp)
        # calculate the derivative for all the vector E 
        elast = zeros_elast(zeros,dmp)
        mat_ze = [zeros elast]
        # check if it is unique 
        # if derivative < 0 return the equilibrium
        if length(zeros) == 1
            if elast[1,1] < 0
                ev = zeros
                return ev
            else 
                ev = minimum(dmp.E)
                return ev[1]
            end
        else 
            elast_neg = mat_ze[all.(<(0), mat_ze[:,2]), :]
            ev = maximum(elast_neg[:,1])
            return ev[1]
        end
    else 
        if dmp.E == []
            ev = 0.001
            return ev
        else 
            ev = minimum(dmp.E)
        return ev[1]
        end
    end
end


function eq_values(dmp)
        # rV! es una función que rellena los vectores vacíos. Entonces le damos un modelo DMP 
    # Ese modelo tiene vectores vacíos de longitud E(vector de estado) que rV! llena
    # para cada E te devuelve un a,α,wv,rvv 
    gmat = zeros(1,10)
    rV!(dmp)
    # take the equilibrium at employment
    ev = selection!(dmp)
    ev = ev[1]
    # evaluate all functions at equilibrium
    av = av!(ev,dmp)
    αv = αv!(ev,dmp)
    wv = wv!(ev,av,αv,dmp)
    # Value functions
    rvv = rvv!(ev,αv,wv,dmp)
    rvf = rvf!(ev,αv,wv,dmp)
    rve = rve!(av,wv,dmp)
    rvu = rvu!(av,wv,dmp)
    # filling the equilibrium matrix
    gmat[1,1] = rvv
    # rVu
    gmat[1,2] = rvu
    # rVe 
    gmat[1,3] = rve
    # rVf 
    gmat[1,4] = rvf
    # Employment 
    gmat[1,5] = ev
    # αv 
    gmat[1,6] = αv
    # av
    gmat[1,7] = av
    # wv 
    gmat[1,8] = wv
    # unemployment 
    gmat[1,9] = dmp.pars[:L] - ev
    return gmat 
end

function rvv_h!(ev,h,dmp::Matching)
    # given an employment level and a 'step' this function returns
    # the value of posting a vacancy evaluated at ev+h
    r,b,Cf,Cv,τw,τu,A,γ,β,L,κ,B = dmp.pars[:r],dmp.pars[:b],dmp.pars[:Cf],dmp.pars[:Cv],dmp.pars[:τw],dmp.pars[:τu],dmp.pars[:A],dmp.pars[:γ],dmp.pars[:β],dmp.pars[:L],dmp.pars[:κ],dmp.pars[:B]
    evh = ev +  h
    rvv = - Cv + (κ^(1/γ) * ((b*evh)^(1-(1/γ))) * ((L - evh)^(β/γ))) * (A - (((r+b+ ((b * evh)/(L - evh))) * (A - Cf + Cv - B * L /evh) + (r + b + (κ^(1/γ) * ((b*evh)^(1-(1/γ))) * ((L - evh)^(β/γ)) ))*(τw - τu))/(2*r + 2*b + ((b * evh)/(L - evh)) + (κ^(1/γ) * ((b*evh)^(1-(1/γ))) * ((L - evh)^(β/γ)) ))) - Cf + Cv - B*L/evh)/(r + b +(κ^(1/γ) * ((b*evh)^(1-(1/γ))) * ((L - evh)^(β/γ)))) 
    return rvv
end

function rvvp!(ev,h,dmp::Matching)
    # given an empleyment level this function calculates the value of posting a vanacy at ev (rvv)
    # and the value of posting a vavancy at ev+h 
    # Then, this function returns 
    αv = αv!(ev,dmp)
    av = av!(ev,dmp)
    wv = wv!(ev,av,αv,dmp)
    rvvh = rvv_h!(ev,h,dmp)
    rvv = rvv!(ev,αv,wv,dmp)
    #= 
    rvv'e = (rvv(e+h)-rvv(e))/h
    =#
    rvvp = (rvvh - rvv)/h
    return rvvp
end




####################
# MODEL STRUCT
####################
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

#
# MODEL PARAMETERS AND VECTORS
#
function dmp(;τw = 3, τu = 1, b=0.2,Cv = 1, 
    Cf = 1, A = 10, γ = 0.5, 
    κ = 1, β = 0.5, r = 0.02, NE = 10, se=0.001, L = NE + 1e-4, B = 0)
    
    pars = Dict(:τw => τw, :τu => τu, 
    :b => b, :Cv=>Cv, :Cf => Cf, :A => A,
    :γ=> γ, :κ => κ, :β => β, :r => r, :NE => NE,:se =>se, :L=>L, :B =>B)
    E = collect(0.001:se:NE)
    agrid = similar(E)
    αgrid=similar(E)
    wgrid = similar(E)
    rVu = similar(E)
    rVv = similar(E)
    rVf = similar(E)
    rVe = similar(E)
    wgrid = similar(E)
    return dmp(pars, E, agrid, αgrid, wgrid, rVv, rVu,rVf,rVe)
end

function q_iter!(lab_descount,q,new_ne,iter,τv)
    model = dmp(;NE = new_ne, B = τv, A = 10,se = 0.001)
    qgmat = eq_values(model)
    ev = qgmat[1,5]
    a = av!(ev,model)
    α = αv!(ev,model)
    w = wv!(ev,a,α,model)
    rvu_ = rvu!(a,w,model)
    rvu_eq = qgmat[1,2]
    if abs(rvu_ - rvu_eq) > 1e-2
        return print("RVU NO DA LO MISMO")
    end
    if rvu_eq < q
        new_ne *= lab_descount
        iter += 1
        Lab = new_ne
        print("
        L iteration $iter, L = $Lab")
        if Lab < 3e-02
            print("
            WARNING: labour market has shrinked")
        end
        return q_iter!(lab_descount,q,new_ne,iter,τv)
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
        return rvv, rvu, rve, rvf, ev, αv, av, wv, lab_force, self_employ, uq,w_net
    end
end

function iter_loop(ft,lab_discount)
    τgrid = collect(0:0.1:ft)
    q_comparatives_x = zeros(14,size(τgrid,1))
    τ_iter = 0
    iter = 0
    new_ne = 10
    Threads.@threads for jτ in eachindex(τgrid)
        iter = 0;
        τ_iter += 1;
        print("
        τ iterations $τ_iter
        ") 
        τv = τgrid[jτ]
        q = 1 + τv
        rvv, rvu, rve, rvf, ev, αv, av, wv, lab_force, self_employ, uq,w_net = q_iter!(lab_discount,q,new_ne,iter,τv)
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
    end
    return q_comparatives_x
end

BIG_loop = iter_loop(3.8,0.99)

τgrid = collect(0:0.1:5);

model = dmp(; τw = 7);
 results_alpha = zeros(size(model.E,1)
 , size(τgrid,1));
 results_rvu = zeros(size(model.E,1)
 , size(τgrid,1));
 results_rvv= zeros(size(model.E,1)
 ,size(τgrid,1));
 results_w= zeros(size(model.E,1)
 , size(τgrid,1));
for jτ in eachindex(τgrid)
    τv = τgrid[jτ]
    model = dmp(;NE = 10, B=τv, A = 10);
    rV!(model);
    results_rvu[:,jτ] = model.rVu;
    results_rvv[:,jτ] = model.rVv;
    results_w[:,jτ] = model.wgrid;
    results_alpha[:,jτ] = model.αgrid
end

plot_rVv = plot([scatter(x= model.E, y=(results_rvv[:,jτ]), 
name = "y = rVv(B = $(τgrid[jτ]))") for jτ in eachindex(τgrid)], 
Layout(title="Value of Posting a Vacancy", xaxis_title="Employment"))

plot_rVu = plot([scatter(x= model.E, y=(results_rvu[:,jτ]), 
name = "y = rVu(B = $(τgrid[jτ]))") for jτ in eachindex(τgrid)], 
Layout(title="", xaxis_title="Employment"))


# LEVEL
self_employment_low_B = 10 .- (BIG_loop[5,:] .+ BIG_loop[9,:]);
BIG_loop[11,:];
employment_B_low = plot([scatter(x= τgrid, y=BIG_loop[5,:], 
name = "E*(B)")], 
Layout(title="Employment level", xaxis_title="B"));

unemployment_B_low = plot([scatter(x= τgrid, y=BIG_loop[9,:], 
    name = "U*(B)")], 
    Layout(title="Unemployment level", xaxis_title="B"));

self_B_low = plot([scatter(x= τgrid, y=self_employment_low_B, 
name = "H*(B)")], 
Layout(title="Self employment level", xaxis_title="B"));

pop_plot_low_B = ([employment_B_low unemployment_B_low self_B_low])
savefig(pop_plot_low_B,"replication_files/new_figures/BIG/level_cut.png")

# WAGE

wage = plot([scatter(x= τgrid, y=BIG_loop[8,:], 
name = "w(B)")], 
Layout(title="w", xaxis_title="B"));
take_home_wage = BIG_loop[8,:] .+ τgrid .- 3 
wage_at_home = plot([scatter(x= τgrid, y=take_home_wage, 
name = "W*(B) + B - τw")], 
Layout(title="Take home wage", xaxis_title="B"));

wage_tot = [wage wage_at_home]
savefig(wage_tot,"replication_files/new_figures/BIG/wages_cut.png")
qgrid = collect(0:0.1:5) .+ 1

rvu = plot([scatter(x= τgrid, y=BIG_loop[2,:], 
name = "rvu(B)")], 
Layout(title="rvu", xaxis_title="B"))

plot_rvu_rvh = plot([scatter(x=τgrid, 
y = BIG_loop[2,:], name="rvu"); scatter(x=τgrid, 
y=qgrid, name="rvh")],
 Layout(title = "rvu,rvh",xaxis_title = "B"))
savefig(plot_rvu_rvh,"replication_files/new_figures/BIG/rvu_rvh.png")

plot_rates_low = plot([scatter(x=τgrid, 
y = BIG_loop[2,:], name="rvu"); scatter(x=τgrid, 
y=qgrid, name="q")],
 Layout(title = "rvu and q, q_start = 1",xaxis_title = "B"))

# RATES

tot_pop_low_s = self_employment_low_B .+ BIG_loop[5,:] .+ BIG_loop[9,:]
tot_pop_low_s = BIG_loop[11,:] .+ BIG_loop[5,:] .+ BIG_loop[9,:]
e_rate_low = BIG_loop[5,:] ./ tot_pop_low_s;
u_rate_low = BIG_loop[9,:] ./ tot_pop_low_s;
self_rate_low = BIG_loop[11,:] ./ tot_pop_low_s
plot_rates_low = plot([scatter(x=τgrid, 
y = e_rate_low, name="E/L"); scatter(x=τgrid, 
y=u_rate_low, name="U/L"); scatter(x=τgrid, 
y=self_rate_low, name="H/L")],
 Layout(title = "Employment and Unemployment rates, q = 1",xaxis_title = "B"))

savefig(plot_rates_low, "replication_files/new_figures/BIG/BIG_rates_cut.png")

# WAGE AND RVU

wage_at_home = plot([scatter(x= τgrid, y=take_home_wage, 
name = "W*(B) + B - τw")], 
Layout(title="Take home wage", xaxis_title="B"))
savefig(wage_at_home, "replication_files/figures/BIG_take_home.png")

wage_B_low = plot([scatter(x= τgrid, y=BIG_loop[8,:], 
name = "W*(B)")], 
Layout(title="Wage", xaxis_title="B"))

rvu_B_low = plot([scatter(x= τgrid, y=BIG_loop[2,:], 
    name = "rVu*(B)")], 
Layout(title="Value of Searching", xaxis_title="B"))
savefig(rvu_B_low, "replication_files/figures/BIG_rvu.png")

rvf_B_low = plot([scatter(x= τgrid, y=BIG_loop[4,:], 
    name = "rVu*(B)")], 
Layout(title="Value of Filling the Vacancy", xaxis_title="B"))

# 
# big * L total amount of subsidies and wv*E total compensation and total income is A*E 
#(BIG * L + wv * E )/(A*E) should be < 1 
# 
τgrid
BIG_loop[:,8]
surplus = (τgrid .* 10 .+ ((BIG_loop[8,:]).*BIG_loop[5,:])) ./ (10 * BIG_loop[5,:])

worker_share =  plot([scatter(x= τgrid, y=surplus, 
    name = "worker share")], 
Layout(title="Worker Share", xaxis_title="B"))
savefig(worker_share, "replication_files/new_figures/BIG/worker_share.png")