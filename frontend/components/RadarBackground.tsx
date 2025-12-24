"use client";

import { useEffect, useState } from "react";

function uniqueId(): number {
    return Math.floor(Math.random() * 1_000_000_000);
}

type Blip = {
    id: number;
    x: number; // %
    y: number; // %
};

export default function RadarBackground() {
    const [blips, setBlips] = useState<Blip[]>([]);

    // Spawn a few dots slowly, then fade them out
    useEffect(() => {
        const interval = window.setInterval(() => {
            const id = uniqueId();

            // scatter around the radar center (50%, 40%) within ~40% radius
            const angle = Math.random() * Math.PI * 2;
            const radius = 10 + Math.random() * 30; // keep mostly inside rings
            const x = 50 + radius * Math.cos(angle);
            const y = 40 + radius * Math.sin(angle);

            setBlips((prev) => {
                const trimmed = prev.length > 7 ? prev.slice(prev.length - 7) : prev; // keep few dots
                return [...trimmed, { id, x, y }];
            });

            window.setTimeout(() => {
                setBlips((prev) => prev.filter((b) => b.id !== id));
            }, 1600);
        }, 2200); // slower dot creation (change to 3000 for even fewer)

        return () => window.clearInterval(interval);
    }, []);

    return (
        <div className="radar-container">
            {/* Soft radar rings */}
            <div className="radar-ping"></div>
            <div className="radar-ping"></div>
            <div className="radar-ping"></div>

            {/* Few occasional blips */}
            {blips.map((b) => (
                <div
                    key={b.id}
                    className="radar-blip"
                    style={{ left: `${b.x}%`, top: `${b.y}%` }}
                />
            ))}
        </div>
    );
}
