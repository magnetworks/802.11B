`timescale 1ns/1ps

module tb_lfsr_scrambler;

  reg clk;
  reg rst_n;
  reg bit_in;
  wire bit_out;

  integer i;
  integer f;

  // DUT
  lfsr_scrambler dut (
    .clk(clk),
    .rst_n(rst_n),
    .bit_in(bit_in),
    .bit_out(bit_out)
  );

  // Clock generator: 10ns period
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  // Stimulus
  initial begin
    rst_n = 0;
    bit_in = 0;
    #20;
    rst_n = 1;

    // open output file
    f = $fopen("scrambler_out.txt", "w");
    if (f == 0) begin
      $display("ERROR: could not open file");
      $finish;
    end

    // feed a 64-bit test pattern
    for (i = 0; i < 64; i = i+1) begin
      bit_in = i[0];  // just alternate 0/1 for testing
      @(posedge clk);
      $fwrite(f, "%0d %0d %0d\n", i, bit_in, bit_out);
    end

    $fclose(f);
    $display("Simulation finished. Results written to scrambler_out.txt");
    $stop;
  end

endmodule
