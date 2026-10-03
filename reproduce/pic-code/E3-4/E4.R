#!/usr/bin/env Rscript
# Figure 8 (table2): threshold sweep vs rrBLUP. 逐行沿用 pic-code/fig8_maize.R, 仅换输入/标题/输出。
# 用法: Rscript fig8_table2.R maize|arab|rice
args <- commandArgs(trailingOnly = TRUE)
ds <- if (length(args) >= 1) args[1] else "maize"
LAB <- list(maize = "Maize MAGIC", arab = "Arabidopsis", rice = "Rice (indica, control)")[[ds]]
HERE <- file.path("reproduce/pic-code/E3-4", ds)
dir.create(HERE, showWarnings = FALSE, recursive = TRUE)
suppressPackageStartupMessages(library(ggplot2))

dat <- read.csv(file.path(HERE, sprintf("fig5_%s_sweep_data.csv", ds)), stringsAsFactors = FALSE)
thr <- sort(unique(dat$Threshold))

rec_d <- dat[dat$Method == "tp1+Meg+Dgp_rec", c("Threshold", "Mean_Diff", "n_traits")]
rec_k <- dat[dat$Method == "tp1+knode_rec", c("Threshold", "Mean_Diff", "n_traits")]
it_d  <- dat[dat$Method == "tp1+Meg+Dgp_iter", c("Threshold", "Mean_Diff", "n_traits")]
it_k  <- dat[dat$Method == "tp1+knode_iter", c("Threshold", "Mean_Diff", "n_traits")]

rec <- merge(rec_d, rec_k, by = "Threshold", suffixes = c(".d", ".k"), all = TRUE)
it  <- merge(it_d, it_k, by = "Threshold", suffixes = c(".d", ".k"), all = TRUE)
rec$xs <- match(rec$Threshold, thr)
it$xs  <- match(it$Threshold, thr)

slot_w <- 0.45
pos_rec <- rec$xs - slot_w / 2
pos_it  <- it$xs + slot_w / 2
BAR_HALF <- 0.18

mk_seg <- function(d, k, pos, protocol) {
  d[is.na(d)] <- 0; k[is.na(k)] <- 0
  lo <- pmin(d, k); hi <- pmax(d, k)
  bot <- ifelse(d <= k, "DynamicGP+MegaLMM+TP1", "Kode")
  top <- ifelse(d <= k, "Kode", "DynamicGP+MegaLMM+TP1")
  data.frame(x = pos, xmin = pos - BAR_HALF, xmax = pos + BAR_HALF,
             ymin = c(rep(0, length(pos)), lo), ymax = c(lo, hi),
             Method = c(bot, top), Protocol = protocol, stringsAsFactors = FALSE)
}
seg <- rbind(mk_seg(rec$Mean_Diff.d, rec$Mean_Diff.k, pos_rec, "rec"),
             mk_seg(it$Mean_Diff.d,  it$Mean_Diff.k,  pos_it,  "iter"))
seg <- seg[seg$ymax - seg$ymin > 1e-6, ]
seg$Group <- paste0(seg$Protocol, "-", seg$Method)
group_levels <- c("iter-DynamicGP+MegaLMM+TP1", "rec-DynamicGP+MegaLMM+TP1", "iter-Kode", "rec-Kode")
seg$Group <- factor(seg$Group, levels = group_levels)

hi_rec <- pmax(rec$Mean_Diff.d, rec$Mean_Diff.k); hi_rec[is.na(hi_rec)] <- 0
hi_it  <- pmax(it$Mean_Diff.d,  it$Mean_Diff.k);  hi_it[is.na(hi_it)]  <- 0
lab_top <- rbind(data.frame(x = pos_rec, y = hi_rec, lab = sprintf("%.2f", hi_rec)),
                 data.frame(x = pos_it,  y = hi_it,  lab = sprintf("%.2f", hi_it)))
lab_top <- lab_top[!is.na(lab_top$y), ]

lo_rec <- pmin(rec$Mean_Diff.d, rec$Mean_Diff.k); lo_rec[is.na(lo_rec)] <- 0
lo_it  <- pmin(it$Mean_Diff.d,  it$Mean_Diff.k);  lo_it[is.na(lo_it)]  <- 0
lab_in <- rbind(
  data.frame(x = pos_rec, y = lo_rec / 2,
             lab = ifelse(rec$Mean_Diff.d <= rec$Mean_Diff.k, rec$n_traits.d, rec$n_traits.k)),
  data.frame(x = pos_it,  y = lo_it / 2,
             lab = ifelse(it$Mean_Diff.d <= it$Mean_Diff.k, it$n_traits.d, it$n_traits.k)),
  data.frame(x = pos_rec, y = lo_rec + (hi_rec - lo_rec) / 2,
             lab = ifelse(rec$Mean_Diff.d <= rec$Mean_Diff.k, rec$n_traits.k, rec$n_traits.d)),
  data.frame(x = pos_it,  y = lo_it + (hi_it - lo_it) / 2,
             lab = ifelse(it$Mean_Diff.d <= it$Mean_Diff.k, it$n_traits.k, it$n_traits.d)))
lab_in <- lab_in[!is.na(lab_in$lab) & !is.na(lab_in$y) & lab_in$y > 0, ]

group_cols <- c("iter-DynamicGP+MegaLMM+TP1" = "#1F77B4", "rec-DynamicGP+MegaLMM+TP1" = "#1F77B4",
                "iter-Kode" = "#D62728", "rec-Kode" = "#D62728")
group_alpha <- c("iter-DynamicGP+MegaLMM+TP1" = 1.0, "rec-DynamicGP+MegaLMM+TP1" = 0.45,
                 "iter-Kode" = 1.0, "rec-Kode" = 0.45)

fig <- ggplot(seg, aes(fill = Group, alpha = Group)) +
  geom_rect(aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)) +
  scale_fill_manual(values = group_cols, drop = FALSE) +
  scale_alpha_manual(values = group_alpha) +
  geom_text(data = lab_top, aes(x = x, y = y, label = lab), size = 2.5, colour = "black",
            vjust = -0.6, inherit.aes = FALSE) +
  geom_label(data = lab_in, aes(x = x, y = y, label = lab), size = 2, colour = "black",
             fill = "white", vjust = 0.5, label.size = 0,
             label.padding = unit(0.1, "lines"), inherit.aes = FALSE) +
  scale_x_continuous(breaks = seq_along(thr), labels = thr,
                     limits = c(0.4, length(thr) + 0.6)) +
  scale_y_continuous(limits = c(0, 1.0)) +
  labs(x = "Predictability threshold", y = "Mean difference in accuracy vs baseline",
       title = sprintf("%s: prediction improvement vs rrBLUP", LAB)) +
  theme_classic(base_family = "Times New Roman", base_size = 8) +
  theme(text = element_text(family = "Times New Roman"),
        legend.position = "top", legend.box = "horizontal", legend.direction = "horizontal",
        legend.key.size = unit(0.3, "cm"),
        axis.title = element_text(size = 8, face = "bold"),
        axis.text = element_text(size = 7, face = "bold", colour = "black"),
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(size = 9, face = "bold", hjust = 0),
        legend.title = element_text(size = 7, face = "bold"),
        legend.text = element_text(size = 7, face = "bold")) +
  guides(fill = guide_legend(title = "Method", nrow = 2, byrow = TRUE, override.aes = list(alpha = group_alpha)),
         alpha = "none")

ggsave(file.path(HERE, sprintf("FigureE4_%s.svg", ds)), fig, device = "svg", width = 200, height = 100, units = "mm")
cat(sprintf("Saved: %s/FigureE4_%s.svg\n", HERE, ds))
