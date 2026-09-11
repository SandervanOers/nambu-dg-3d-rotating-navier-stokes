function ops = build_operators_DG0_periodic(Nel)
%BUILD_OPERATORS_DG0_PERIODIC Compatibility wrapper for periodic DG(0).
ops = build_operators_DG0(Nel, [false false false]);
end
