use crate::nn::{NeuralNetwork, Layer};
use crate::activation::ActivationFunction;
use std::fmt;

impl<A> fmt::Display for NeuralNetwork<f64, A>
where
    A: ActivationFunction<f64>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.print_network(f)
    }
}

impl<A> NeuralNetwork<f64, A>
where
    A: ActivationFunction<f64>,
{
    pub fn print_network(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║                               NEURAL NETWORK                                    ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════════════════╣")?;
        
        // Network overview
        writeln!(f, "║ Architecture: {:>60} ║", 
            self.architecture.iter()
                .map(|&x| x.to_string())
                .collect::<Vec<_>>()
                .join(" → ")
        )?;
        
        writeln!(f, "║ Total Layers: {:>8} │ Parameters: {:>12} │ Concurrent: {:>8} ║",
            self.num_layers(),
            self.parameter_count(),
            if self.is_concurrent() { "Yes" } else { "No" }
        )?;
        
        writeln!(f, "╠══════════════════════════════════════════════════════════════════════════════╣")?;
        
        // Layer details
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "║ Layer {} │ Input: {:>3} │ Output: {:>3} │ Activation: {:>12} │ Params: {:>6} ║",
                i + 1,
                layer.input_size(),
                layer.output_size(),
                layer.activation.name(),
                layer.weights.rows() * layer.weights.cols() + layer.biases.rows()
            )?;
            
            if i < self.layers.len() - 1 {
                writeln!(f, "║{:^78}║", "│")?;
            }
        }
        
        writeln!(f, "╚══════════════════════════════════════════════════════════════════════════════╝")?;
        
        Ok(())
    }

    pub fn print_detailed(&self) {
        println!("{}", self);
        
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
            println!("║                              LAYER {} DETAILS                                  ║", layer_idx + 1);
            println!("╠══════════════════════════════════════════════════════════════════════════════╣");
            
            // Print weight matrix
            println!("║ Weights Matrix ({}×{}):", layer.weights.rows(), layer.weights.cols());
            self.print_matrix_preview(&layer.weights, "║ ");
            
            println!("║");
            
            // Print bias vector
            println!("║ Bias Vector ({}×{}):", layer.biases.rows(), layer.biases.cols());
            self.print_matrix_preview(&layer.biases, "║ ");
            
            println!("╚══════════════════════════════════════════════════════════════════════════════╝");
        }
    }

    fn print_matrix_preview(&self, matrix: &matrix::Matrix<f64>, prefix: &str) {
        let (rows, cols) = matrix.dimensions();
        let max_display_rows = 5;
        let max_display_cols = 8;
        
        for i in 0..rows.min(max_display_rows) {
            print!("{}", prefix);
            for j in 0..cols.min(max_display_cols) {
                if let Ok(val) = matrix.get(i, j) {
                    print!("{:>8.4} ", val);
                }
            }
            if cols > max_display_cols {
                print!("...");
            }
            println!();
        }
        
        if rows > max_display_rows {
            println!("{}  ⋮", prefix);
        }
    }
}