# Independent SI reconstruction for the pinned series-only, four-wire feeder.
# Reject unsupported topology/device changes rather than silently omitting them.
function check_fixture_physics(net, result)
    @assert Set(keys(net["bus"])) == Set(["source", "near", "joint", "far"])
    @assert length(net["line"]) == 3 && length(net["load"]) == 2
    terminals = ["1", "2", "3", "n"]
    voltage(bus) = [
        complex(result["bus"][bus][t]["vr"], result["bus"][bus][t]["vi"]) for t in terminals
    ]
    balance = Dict(bus => zeros(ComplexF64, 4) for bus in keys(net["bus"]))
    current_error = 0.0
    max_thermal_violation = 0.0
    reconstructed_loss = 0.0
    for (id, line) in net["line"]
        @assert line["terminal_map_from"] == line["terminal_map_to"] == terminals
        code = net["linecode"][line["linecode"]]
        Z = [
            complex(
                code["R_series_$(min(i,j))_$(max(i,j))"],
                code["X_series_$(min(i,j))_$(max(i,j))"],
            ) * line["length"] for i = 1:4, j = 1:4
        ]
        from, to = line["bus_from"], line["bus_to"]
        drop = voltage(from) - voltage(to)
        current = Z \ drop
        @assert all(isfinite, current)
        balance[from] .+= current
        balance[to] .-= current
        reconstructed_loss += real(sum(drop .* conj.(current)))
        max_thermal_violation =
            max(max_thermal_violation, maximum(abs.(current) - code["i_max"]))
        for (k, t) in enumerate(terminals)
            entry = result["line"][id][t]
            returned_from = complex(entry["cr_fr"], entry["ci_fr"])
            returned_to = complex(entry["cr_to"], entry["ci_to"])
            current_error = max(
                current_error,
                abs(returned_from - current[k]),
                abs(returned_to + current[k]),
            )
        end
    end
    for load in values(net["load"])
        @assert load["configuration"] == "WYE" && load["model"] == "constant_power"
        @assert load["terminal_map"] == terminals
        bus = load["bus"]
        V = voltage(bus)
        current = conj.(complex.(load["p_nom"], load["q_nom"]) ./ (V[1:3] .- V[4]))
        balance[bus][1:3] .+= current
        balance[bus][4] -= sum(current)
    end
    # The source supplies its own unconstrained currents, including ground return.
    kcl = maximum(maximum(abs, balance[bus]) for bus in ("near", "joint", "far"))
    max_voltage_violation = 0.0
    for (bus, data) in net["bus"]
        haskey(data, "vpn_min") || continue
        V = voltage(bus)
        magnitudes = abs.(V[1:3] .- V[4])
        max_voltage_violation = max(
            max_voltage_violation,
            maximum(data["vpn_min"] - magnitudes),
            maximum(magnitudes - data["vpn_max"]),
        )
    end
    source = net["voltage_source"]["grid"]
    expected = source["v_magnitude"] .* cis.(source["v_angle"])
    source_error =
        max(maximum(abs, voltage("source")[1:3] - expected), abs(voltage("source")[4]))
    @assert kcl <= 1e-5 kcl
    @assert current_error <= 1e-5 current_error
    @assert max_thermal_violation <= 1e-5 max_thermal_violation
    @assert max_voltage_violation <= 1e-5 max_voltage_violation
    @assert source_error <= 1e-5 source_error
    @assert isfinite(reconstructed_loss) && reconstructed_loss >= -1e-8
    return Dict(
        "max_kcl_ampere" => kcl,
        "max_branch_current_error_ampere" => current_error,
        "max_thermal_violation_ampere" => max_thermal_violation,
        "max_voltage_violation_volt" => max_voltage_violation,
        "source_voltage_error_volt" => source_error,
        "reconstructed_loss_watt" => reconstructed_loss,
        "current_tolerance_ampere" => 1e-5,
        "voltage_tolerance_volt" => 1e-5,
    )
end
